const express = require("express");
const bodyParser = require("body-parser");

const app = express();
const PORT = 8000;

const cors = require("cors");
app.use(cors());

app.use(bodyParser.json());

// Parameters and state
let client = null;
let server = null;

// Helper functions
function discreteGaussianSample(mean, stdDev, size) {
  const samples = Array(size)
    .fill()
    .map(() => Math.round(mean + stdDev * (Math.random() - 0.5)));
  return samples;
}

function addPolynomials(poly1, poly2, modulus) {
  return poly1.map((a, i) => (a + poly2[i]) % modulus);
}

function multiplyPolynomials(poly1, poly2, modulus) {
  const degree = poly1.length;
  const result = Array(degree * 2).fill(0);

  for (let i = 0; i < degree; i++) {
    for (let j = 0; j < degree; j++) {
      result[i + j] = (result[i + j] + poly1[i] * poly2[j]) % modulus;
    }
  }

  // Reduce modulo X^degree + 1
  for (let i = degree; i < result.length; i++) {
    result[i - degree] = (result[i - degree] - result[i]) % modulus;
  }

  return result.slice(0, degree);
}

function respond(cX) {
  const { modulus, noiseStdDev, k } = server;
  const eS = discreteGaussianSample(0, noiseStdDev, k.length);
  const dX = addPolynomials(multiplyPolynomials(cX, k, modulus), eS, modulus);
  return { dX };
}

function finalize() {
  const { modulus, state } = client;
  const { c } = server;
  const { s } = state;

  const p = 65536; // Scaling factor
  const term = addPolynomials(
    server.c,
    state.s.map((coeff) => -coeff % modulus),
    modulus
  );

  const yX = term.map((coeff) => Math.floor((p * coeff) / modulus));
  return { yX };
}

// Routes
// State initialization
app.post("/initialize", (req, res) => {
    try {
      console.log("Initializing client and server...");
      
      // Your existing initialization code
      const degree = 64;
      const modulus = 65537;
      const noiseStdDev = 3.2;
  
      const a = Array(degree).fill(1);
      const a0 = Array(degree).fill(2);
      const a1 = Array(degree).fill(3);
  
      client = { degree, modulus, noiseStdDev, a, a0, a1 };
  
      console.log("Client initialized:", client);  // Log client initialization
  
      // Server setup (example)
      const k = discreteGaussianSample(0, noiseStdDev, degree);
      const e = discreteGaussianSample(0, noiseStdDev, degree);
      const c = addPolynomials(multiplyPolynomials(a, k, modulus), e, modulus);
  
      server = { degree, modulus, noiseStdDev, a, k, c };
  
      res.json({ message: "Initialized client and server." });
    } catch (error) {
      console.error("Error initializing:", error);
      res.status(500).json({ error: "Failed to initialize client and server." });
    }
  });
  

 app.post("/query", (req, res) => {
    try {
      const { x } = req.body;
  
      if (!client) {
        throw new Error("Client is not initialized. Call /initialize first.");
      }
  
      const { degree, modulus, noiseStdDev, a, a0, a1 } = client;
  
      // Generate Gaussian noise samples
      const s = discreteGaussianSample(0, noiseStdDev, degree);
      const eC = discreteGaussianSample(0, noiseStdDev, degree);
  
      // Compute the hashed representation a_F(x)
      let aF = x.split("").reduce((acc, bit) => {
        return multiplyPolynomials(acc, bit === "1" ? a1 : a0, modulus);
      }, a0);
  
      // Calculate cX (blinded query)
      const cX = addPolynomials(
        addPolynomials(multiplyPolynomials(a, s, modulus), eC, modulus),
        aF,
        modulus
      );
  
      // Store the state for the finalize step
      client.state = { s, eC };
  
      // Return the response
      res.json({ cX });
    } catch (error) {
      console.error("Error during query:", error);
      res.status(500).json({ error: error.message });
    }
  });
  
  

app.post("/respond", (req, res) => {
  const { cX } = req.body;
  const result = respond(cX);
  res.json(result);
});

app.post("/finalize", (req, res) => {
  const result = finalize();
  res.json(result);
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
