const PredictionResult = ({ prediction }) => (
  <div className="prediction-result">
  <p id="result" className={prediction ? "result-box" : "placeholder"}>
    {prediction
      ? `Estimated AC Power Output: ${prediction} kW`
      : "Enter solar plant data above to get AC power prediction."}
  </p>

  {prediction === "Error connecting to backend!" && (
    <p className="error-box">
      {prediction} (check that the ML service on port 5001 is running)
    </p>
  )}
</div>

);

export default PredictionResult;
