import { useState } from 'react'
import axios from 'axios'
import Cropper from 'react-easy-crop'
import ImageCropper from './components/ImageCropper'
import { getCroppedImg } from './utils/cropUtils'
import './App.css'

function App() {
  const [apImage, setApImage] = useState(null)
  const [latImage, setLatImage] = useState(null)

  const [apCropPixel, setApCropPixel] = useState(null)
  const [latCropPixel, setLatCropPixel] = useState(null)

  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [selectedResult, setSelectedResult] = useState(null)
  const [modalImage, setModalImage] = useState(null)
  const [modalZoom, setModalZoom] = useState(1)
  const [modalCrop, setModalCrop] = useState({ x: 0, y: 0 }) // Required by Cropper
  const [error, setError] = useState(null)

  const handleFileChange = (e, setImg) => {
    if (e.target.files && e.target.files.length > 0) {
      const reader = new FileReader()
      reader.addEventListener('load', () => setImg(reader.result))
      reader.readAsDataURL(e.target.files[0])
    }
  }


  const analyze = async () => {
    if (!apImage || !latImage || !apCropPixel || !latCropPixel) return

    setLoading(true)
    setError(null)
    setResult(null)
    setSelectedResult(null)

    try {
      const apBlob = await getCroppedImg(apImage, apCropPixel)
      const latBlob = await getCroppedImg(latImage, latCropPixel)

      const formData = new FormData()
      formData.append('ap_image', apBlob, 'ap.jpg')
      formData.append('lat_image', latBlob, 'lat.jpg')

      // Use environment variable or default to localhost
      // For testing on same machine use localhost:8000
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

      const res = await axios.post(`${API_URL}/predict`, formData)
      setResult(res.data)
      // Default to the top prediction (first in list)
      if (res.data.results && res.data.results.length > 0) {
        setSelectedResult(res.data.results[0])
      }

    } catch (err) {
      console.error(err)
      setError("Analysis failed. Ensure backend is running.")
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setApImage(null)
    setLatImage(null)
    setResult(null)
    setSelectedResult(null)
  }

  return (
    <div className="app-container">
      <header>
        <div className="logo-container">
          <img src="/logo.png" alt="OrthoScrew Logo" className="app-logo" />
          <h1><span>OrthoScrew</span> ID</h1>
        </div>
      </header>

      {!result && (
        <div className="input-section">
          {/* AP Image Upload/Crop */}
          <div className="view-block">
            {!apImage ? (
              <div className="upload-box">
                <label>AP View</label>
                <label className="custom-file-upload">
                  Choose File
                  <input type="file" accept="image/*" onChange={(e) => handleFileChange(e, setApImage)} />
                </label>
              </div>
            ) : (
              <ImageCropper image={apImage} label="Crop AP Screw" onCropChange={setApCropPixel} />
            )}
          </div>

          {/* Lateral Image Upload/Crop */}
          <div className="view-block">
            {!latImage ? (
              <div className="upload-box">
                <label>Lateral View</label>
                <label className="custom-file-upload">
                  Choose File
                  <input type="file" accept="image/*" onChange={(e) => handleFileChange(e, setLatImage)} />
                </label>
              </div>
            ) : (
              <ImageCropper image={latImage} label="Crop Lateral Screw" onCropChange={setLatCropPixel} />
            )}
          </div>

          {/* Replaced the original 'actions' div and its content */}
          <div className="process-actions">
            <button
              className="process-button"
              onClick={analyze}
              disabled={loading}
            >
              {loading ? 'Processing...' : 'Identify Manufacturer'} {/* Changed text */}
            </button>
            {(apImage || latImage) && <button className="reset-btn" onClick={reset}>Reset</button>}
          </div>

          {error && <div className="error">{error}</div>}

          <div className="disclaimer">
            <p>
              <strong>Disclaimer:</strong> This tool is provided for assistive use only and does not constitute medical advice or diagnosis.
              AI predictions may contain errors. The operating physician assumes full responsibility for the final identification of any implant
              and must independently verify results against official manufacturer references.
            </p>
          </div>
        </div>
      )}

      {result && selectedResult && (
        <div className="result-section">
          <div className="reset-container">
            <button className="reset-btn" onClick={reset}>Start Over</button>
          </div>

          <h3>Predicted Manufacturers</h3>
          <div className="prediction-list">
            {result.results.map((item, idx) => (
              <div
                key={idx}
                className={`prediction-item ${selectedResult.manufacturer === item.manufacturer ? 'selected' : ''}`}
                onClick={() => setSelectedResult(item)}
              >
                <div className="pred-info">
                  <span className="pred-name">{item.manufacturer}</span>
                  <span className="pred-conf">{(item.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="progress-bar-bg">
                  <div className="progress-bar-fill" style={{ width: `${item.confidence * 100}%` }}></div>
                </div>
              </div>
            ))}
          </div>

          <div className="similar-section">
            <h3>Similar Cases ({selectedResult.manufacturer})</h3>
            <div className="similar-list">
              {selectedResult.similar && selectedResult.similar.length > 0 ? (
                selectedResult.similar.map((item, idx) => (
                  <div key={idx} className="similar-item">
                    <div className="similar-imgs">
                      <img
                        src={`http://localhost:8000${item.ap_url}`}
                        alt="Sim AP"
                        onClick={() => { setModalImage(`http://localhost:8000${item.ap_url}`); setModalZoom(1); }}
                      />
                      <img
                        src={`http://localhost:8000${item.lat_url}`}
                        alt="Sim Lat"
                        onClick={() => { setModalImage(`http://localhost:8000${item.lat_url}`); setModalZoom(1); }}
                      />
                    </div>
                    <div className="similar-label">{item.manufacturer}</div>
                    <div className="similar-score">Sim: {item.score.toFixed(3)}</div>
                  </div>
                ))
              ) : (
                <p>No similar examples found in index for this manufacturer.</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Image Modal */}
      {modalImage && (
        <div className="modal-overlay" onClick={() => setModalImage(null)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setModalImage(null)}>×</button>
            <div className="modal-cropper-wrapper">
              <Cropper
                image={modalImage}
                crop={modalCrop}
                zoom={modalZoom}
                onCropChange={setModalCrop}
                onZoomChange={setModalZoom}
                showGrid={false}
                restrictPosition={false}
                style={{ containerStyle: { background: 'rgba(0,0,0,0.8)' } }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
