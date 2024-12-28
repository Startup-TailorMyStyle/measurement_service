import { useState } from 'react';
import './App.css';

function App() {
  const [frontImage, setFrontImage] = useState(null);
  const [sideImage, setSideImage] = useState(null);
  const [height, setHeight] = useState('');
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [measurements, setMeasurements] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!frontImage || !sideImage || !height) {
      setMessage('Please select both images and enter height');
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append('front_image', frontImage);
    formData.append('side_image', sideImage);
    formData.append('height', height);

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      if (data.measurements) {
        setMeasurements(data.measurements);
      }
      setMessage(data.message || data.error);
    } catch (error) {
      setMessage('Error uploading files');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Image Upload</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Front Image:</label>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setFrontImage(e.target.files[0])}
          />
        </div>
        <div>
          <label>Side Image:</label>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setSideImage(e.target.files[0])}
          />
        </div>
        <div>
          <label>Height (cm):</label>
          <input
            type="number"
            value={height}
            onChange={(e) => setHeight(e.target.value)}
          />
        </div>
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Processing...' : 'Upload'}
        </button>
      </form>
      {isLoading && <div className="loader">Processing images...</div>}
      {message && <p>{message}</p>}
      {measurements && (
        <div>
          <h2>Measurements:</h2>
          <p>Hips Circumference: {measurements.hips_circumference.toFixed(2)} cm</p>
          <p>Waist Circumference: {measurements.waist_circumference.toFixed(2)} cm</p>
          <p>Bust Circumference: {measurements.bust_circumference.toFixed(2)} cm</p>
          <p>Biceps Circumference: {measurements.biceps_circumference.toFixed(2)} cm</p>
        </div>
      )}
    </div>
  );
}

export default App;