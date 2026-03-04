import React from "react";

export default function About() {
  return (
    <div className="page page-about">
      <h1>About this project</h1>

      <p>
        EcoCharge is an intelligent solar power forecasting platform that uses 
        machine learning to predict AC power output in solar plants.
      </p>

      <p>
        Accurate energy forecasting is critical for grid stability, load management, 
        and maximizing renewable energy efficiency. EcoCharge analyzes both real-time 
        and historical parameters such as solar irradiance, ambient temperature, 
        previous hour power output, previous day generation trends, and rolling 
        average performance patterns to generate accurate predictions.
      </p>

      <p>
        The backend prediction engine is powered by a Random Forest Regressor 
        model deployed as a Flask microservice. The model is integrated with a 
        responsive React frontend through REST APIs, creating a seamless 
        full-stack machine learning workflow.
      </p>

      <p>
        This project demonstrates frontend development using React (HTML/CSS/JS), 
        backend API integration, machine learning model deployment, and real-time 
        prediction handling within a production-style architecture.
      </p>

      <p>
        EcoCharge bridges data science and full-stack engineering to showcase how 
        AI can enhance renewable energy planning and sustainable power management.
      </p>
    </div>
  );
}
