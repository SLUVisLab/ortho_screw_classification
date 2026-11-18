from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split

from dual_view_dinov2 import model
from dual_view_dinov2.train import collate_valid, prepare_data


def tiny_backbone(**kwargs):
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 384))


class CheckpointContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    @patch.object(model, "dinov2_vits14", side_effect=tiny_backbone)
    @patch.object(torch.hub, "download_url_to_file", side_effect=AssertionError("Inference must be offline"))
    def test_legacy_and_annotated_checkpoint_predictions_match(self, download, backbone):
        original = model.DualViewDINOv2(initialize_backbone=False).eval()
        ap, lateral = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
        with torch.inference_mode():
            expected = original(ap, lateral)
        checkpoints = [original.state_dict(), {"model_state_dict": original.state_dict(), "config": {"classes": model.CLASSES}}]
        for index, checkpoint in enumerate(checkpoints):
            path = self.root / f"checkpoint-{index}.pth"
            torch.save(checkpoint, path)
            loaded = model.load_checkpoint(path)
            with torch.inference_mode():
                torch.testing.assert_close(loaded(ap, lateral), expected)
        download.assert_not_called()

    @patch.object(model, "dinov2_vits14", side_effect=tiny_backbone)
    def test_backbone_frozen_but_classifier_updates(self, backbone):
        network = model.DualViewDINOv2(initialize_backbone=False).train()
        before = network.classifier[0].weight.detach().clone()
        optimizer = torch.optim.Adam(network.classifier.parameters(), lr=1e-3)
        loss = nn.functional.cross_entropy(network(torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)), torch.tensor([0, 1]))
        loss.backward()
        optimizer.step()
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in network.backbone.parameters()))
        self.assertFalse(torch.equal(before, network.classifier[0].weight))

    def test_incompatible_view_and_class_order_fail(self):
        path = self.root / "incompatible.pth"
        torch.save({"classifier.0.weight": torch.zeros(512, 384)}, path)
        with self.assertRaisesRegex(ValueError, "dual-view"):
            model.load_checkpoint(path)
        torch.save({"model_state_dict": {}, "config": {"classes": list(reversed(model.CLASSES))}}, path)
        with self.assertRaisesRegex(ValueError, "class order"):
            model.load_checkpoint(path)

    def test_backbone_checksum_rejects_corruption(self):
        path = self.root / "corrupted.pth"
        path.write_bytes(b"not the pretrained backbone")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            model.pretrained_backbone_path(path)


class DataContractTests(unittest.TestCase):
    def test_patient_split_modes_and_no_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            # Deliberately scramble rows so first-seen patient order differs
            # from groupby order. Each manufacturer has ten patients.
            patients = np.random.default_rng(7).permutation(40)
            rows = [dict(manufacturer=model.CLASSES[int(pid) // 10], patient_number=int(pid),
                         view_position=view, relative_file_path=f"synthetic/images/{pid}_{view}.png")
                    for pid in patients for view in ("AP", "LATERAL")]
            frame = pd.DataFrame(rows)
            frame.to_csv(Path(directory) / "metadata_summary.csv", index=False)
            for mode in ("historical", "aligned"):
                _, _, train_ids, evaluation_ids = prepare_data(directory, mode)
                self.assertFalse(set(train_ids) & set(evaluation_ids))
                self.assertEqual(len(train_ids), 32)
                self.assertEqual(len(evaluation_ids), 8)
                labels = frame.groupby("patient_number").manufacturer.first()
                if mode == "aligned":
                    labels = labels.reindex(patients)
                expected_train, expected_eval = train_test_split(patients, test_size=.2, random_state=42, stratify=labels)
                np.testing.assert_array_equal(train_ids, expected_train)
                np.testing.assert_array_equal(evaluation_ids, expected_eval)

    def test_invalid_samples_are_removed_without_empty_batch_failure(self):
        self.assertIsNone(collate_valid([None, None]))
        sample = (torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), 2)
        ap, lateral, labels = collate_valid([None, sample])
        self.assertEqual(tuple(ap.shape), (1, 3, 224, 224))
        self.assertEqual(tuple(lateral.shape), (1, 3, 224, 224))
        self.assertEqual(labels.tolist(), [2])


if __name__ == "__main__":
    unittest.main()
