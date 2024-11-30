import React, { useState } from "react";
import axios from "axios";
import './App.css';

function App() {
  const [cx, setCx] = useState(null);
  const [dx, setDx] = useState(null);
  const [prfOutput, setPrfOutput] = useState(null);
  const [x, setX] = useState('');
  const [message, setMessage] = useState('');

  // Initialize the client and server
  const initialize = async () => {
    try {
      const response = await axios.post("http://localhost:8000/initialize");
      setMessage({ text: response.data.message, type: "success" });
    } catch (error) {
      setMessage({ text: "Failed to initialize client and server.", type: "error" });
    }
  };

  // Send query to the server
  const sendQuery = async () => {
    try {
      const response = await axios.post("http://localhost:8000/query", { x });
      setCx(response.data.cX);
      setMessage({ text: "Query response received!", type: "success" });
    } catch (error) {
      setMessage({ text: "Error during query.", type: "error" });
    }
  };

  // Send response to the server
  const sendResponse = async () => {
    try {
      const response = await axios.post("http://localhost:8000/respond", { cX: cx });
      setDx(response.data.dX);
      setMessage({ text: "Server response received!", type: "success" });
    } catch (error) {
      setMessage({ text: "Error during response.", type: "error" });
    }
  };

  // Finalize the PRF
  const finalize = async () => {
    try {
      const response = await axios.post("http://localhost:8000/finalize");
      setPrfOutput(response.data.yX);
      setMessage({ text: "PRF output received!", type: "success" });
    } catch (error) {
      setMessage({ text: "Error during finalization.", type: "error" });
    }
  };

  return (
    <div className="app-container">
      <h1>Lattice-Based PRF Simulator</h1>

      {message && (
        <div className={`alert ${message.type}`}>
          {message.text}
        </div>
      )}

      <div>
        <button onClick={initialize}>Initialize</button>
      </div>

      <div>
        <input
          type="text"
          placeholder="Enter binary string (x)"
          value={x}
          onChange={(e) => setX(e.target.value)}
        />
        <button onClick={sendQuery} disabled={!x}>
          Send Query
        </button>
      </div>

      {cx && (
        <div>
          <h3>Query Response (cX):</h3>
          <div className="horizontal-list">
            {cx.map((item, index) => (
              <span key={index}>{item}</span>
            ))}
          </div>
          <button onClick={sendResponse}>Get Server Response</button>
        </div>
      )}

      {dx && (
        <div>
          <h3>Server Response (dX):</h3>
          <div className="horizontal-list">
            {dx.map((item, index) => (
              <span key={index}>{item}</span>
            ))}
          </div>
          <button onClick={finalize}>Finalize PRF</button>
        </div>
      )}

      {prfOutput && (
        <div>
          <h3>PRF Output (yX):</h3>
          <div className="horizontal-list">
            {prfOutput.map((item, index) => (
              <span key={index}>{item}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
