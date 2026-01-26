import { useState } from 'react'
import axios from 'axios'
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
  }

  return (
    <div className="app-container">
      <header>
        <h1>OrthoScrew ID</h1>
      </header>

      {!result && (
        <div className="input-section">
          {/* AP Image Upload/Crop */}
          <div className="view-block">
            {!apImage ? (
              <div className="upload-box">
                <label>Upload AP View</label>
                <input type="file" accept="image/*" onChange={(e) => handleFileChange(e, setApImage)} />
              </div>
            ) : (
              <ImageCropper image={apImage} label="Crop AP Screw" onCropChange={setApCropPixel} />
            )}
          </div>

          {/* Lateral Image Upload/Crop */}
          <div className="view-block">
            {!latImage ? (
              <div className="upload-box">
                <label>Upload Lateral View</label>
                <input type="file" accept="image/*" onChange={(e) => handleFileChange(e, setLatImage)} />
              </div>
            ) : (
              <ImageCropper image={latImage} label="Crop Lateral Screw" onCropChange={setLatCropPixel} />
            )}
          </div>

          <div className="actions">
            <button
              className="analyze-btn"
              disabled={!apImage || !latImage || loading}
              onClick={analyze}
            >
              {loading ? 'Analyzing...' : 'Identify Screw'}
            </button>
            {(apImage || latImage) && <button className="reset-btn" onClick={reset}>Reset</button>}
          </div>

          {error && <div className="error">{error}</div>}
        </div>
      )}

      {result && (
        <div className="result-section">
          <div className="prediction-card">
            <h2>{result.prediction}</h2>
            <div className="confidence">Confidence: {(result.confidence * 100).toFixed(1)}%</div>
            <button className="reset-btn" onClick={reset}>Start Over</button>
          </div>

          <h3>Why? Similar Cases:</h3>
          <div className="similar-list">
            {result.similar && result.similar.map((item, idx) => (
              <div key={idx} className="similar-item">
                <div className="similar-imgs">
                  <img src={`http://localhost:8000${item.ap_url}`} alt="Sim AP" />
                  <img src={`http://localhost:8000${item.lat_url}`} alt="Sim Lat" />
                </div>
                <div className="similar-label">{item.manufacturer}</div>
                <div className="similar-score">Sim: {item.score.toFixed(3)}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default App
