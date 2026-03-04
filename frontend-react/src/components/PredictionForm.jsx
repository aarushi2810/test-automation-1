import { useState } from "react";

const PredictionForm = ({ setPrediction }) => {
  const [formData, setFormData] = useState({
    irradiance: "",
    temp: "",
    prevHour: "",
    prevDay: "",
    roll3: "",
    roll6: "",
  });

  const [errors, setErrors] = useState({});
  const [isLoading, setIsLoading] = useState(false);


  const validateField = (name, value) => {
    const numValue = parseFloat(value);
    if (isNaN(numValue) || value === "") {
      return `${name} is required`;
    }

    switch (name) {
      case "irradiance":
        if (numValue < 0 || numValue > 1000) {
          return "Irradiance must be between 0 and 1000 W/m²";
        }
        break;
      case "temp":
        if (numValue < -20 || numValue > 60) {
          return "Temperature must be between -20 and 60°C";
        }
        break;
      case "prevHour":
      case "prevDay":
      case "roll3":
      case "roll6":
        if (numValue < 0 || numValue > 10) {
          return `${name} must be between 0 and 10 kW`;
        }
        break;
      default:
        return "";
    }
    return "";
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
    
   
    if (errors[name]) {
      setErrors({ ...errors, [name]: "" });
    }
  };

  const handleBlur = (e) => {
    const { name, value } = e.target;
    const error = validateField(name, value);
    setErrors({ ...errors, [name]: error });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    
    const newErrors = {};
    let hasErrors = false;
    
    Object.keys(formData).forEach((key) => {
      const error = validateField(key, formData[key]);
      if (error) {
        newErrors[key] = error;
        hasErrors = true;
      }
    });

    if (hasErrors) {
      setErrors(newErrors);
      return;
    }

    setIsLoading(true);
    try {
      const res = await fetch("http://localhost:5001/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!res.ok) {
        throw new Error("Prediction request failed");
      }

      const result = await res.json();
      const predictedPower = parseFloat(result.predictedPower).toFixed(2);
      setPrediction(predictedPower);
      localStorage.setItem("lastPrediction", result.predictedPower);
    } catch (err) {
      console.error(err);
      setPrediction("Error connecting to backend!");
      setErrors({ submit: "Failed to get prediction. Please check if backend services are running." });
    } finally {
      setIsLoading(false);
    }
  };

  const fieldLabels = {
    irradiance: "Irradiance (W/m²)",
    temp: "Temperature (°C)",
    prevHour: "Previous Hour (kW)",
    prevDay: "Previous Day (kW)",
    roll3: "Rolling 3-Hour Avg (kW)",
    roll6: "Rolling 6-Hour Avg (kW)",
  };

  return (
    <form onSubmit={handleSubmit} className="prediction-form">
      
      {Object.keys(formData).map((key) => (
        <div className="form-group" key={key}>
          <label htmlFor={key}>{fieldLabels[key]}</label>
          <input
            type="number"
            id={key}
            name={key}
            value={formData[key]}
            onChange={handleChange}
            onBlur={handleBlur}
            step="0.01"
            className={errors[key] ? "error" : ""}
            placeholder={`Enter ${fieldLabels[key]}`}
          />
          {errors[key] && <span className="error-message">{errors[key]}</span>}
        </div>
      ))}
      {errors.submit && <div className="error-message submit-error">{errors.submit}</div>}
      <button id="submit" type="submit" disabled={isLoading} className="submit-button">
        {isLoading ? "Predicting..." : "Predict AC Power Output"}
      </button>
    </form>
  );
};

export default PredictionForm;
