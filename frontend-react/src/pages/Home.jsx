// src/pages/Home.jsx
import React from "react";
import "../styles/page.css";

export default function Home() {
  return (
    <div className="page page-home">
      <div className="hero-section">
        <div className="hero-text">
          <h1>Welcome to EcoCharge</h1>
          <p className="lead">Community solar power prediction — small, clear demo my idea.</p>
          <p className="lead">Use the navigation to go to <strong>Predict</strong> and try the model.</p>
        </div>

        <div aria-hidden className="hero-media">
          <img src="/assets/banner.jpg" className="hero-img" alt="solar panels banner" />
        </div>
      </div>
    </div>
  );
}