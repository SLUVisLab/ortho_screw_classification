import React, { useState, useCallback } from 'react'
import Cropper from 'react-easy-crop'
import './ImageCropper.css'

const ImageCropper = ({ image, onCropChange, label }) => {
    const [crop, setCrop] = useState({ x: 0, y: 0 })
    const [zoom, setZoom] = useState(1)

    const onCropComplete = useCallback((croppedArea, croppedAreaPixels) => {
        onCropChange(croppedAreaPixels)
    }, [onCropChange])

    return (
        <div className="cropper-container">
            <h3>{label}</h3>
            <div className="crop-area">
                <Cropper
                    image={image}
                    crop={crop}
                    zoom={zoom}
                    maxZoom={10}
                    aspect={1}
                    onCropChange={setCrop}
                    onCropComplete={onCropComplete}
                    onZoomChange={setZoom}
                />
            </div>
            <div className="controls">
                <label>Zoom</label>
                <input
                    type="range"
                    value={zoom}
                    min={1}
                    max={10}
                    step={0.1}
                    aria-labelledby="Zoom"
                    onChange={(e) => setZoom(e.target.value)}
                    className="zoom-range"
                />
            </div>
        </div>
    )
}

export default ImageCropper
