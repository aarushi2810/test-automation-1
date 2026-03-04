import React, { useState } from "react";
import "../styles/page.css";

export default function Contact() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    apartment: "",
    units: "",
    message: ""
  });

  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const res = await fetch("http://localhost:5001/contact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
      });

      const data = await res.json();
      alert(data.message || "Request submitted successfully!");
      setFormData({
        name: "",
        email: "",
        apartment: "",
        units: "",
        message: ""
      });
    } catch (err) {
      alert("Failed to submit request.");
    } finally {
      setLoading(false);
    }
  };

  return (
  <div className="page-contact">

    <div className="contact-container">

      <div className="contact-hero">
        <h1>Solar Power Prediction for Apartments & Societies</h1>
        <p>
          We provide AI-powered solar output analytics for residential and
          commercial buildings. Get accurate energy forecasting and optimize
          your solar efficiency.
        </p>
      </div>

      <div className="contact-card">
        <h2>Request a Service Consultation</h2>

        <form onSubmit={handleSubmit} className="contact-form">
          <input
            name="name"
            placeholder="Full Name"
            value={formData.name}
            onChange={handleChange}
            required
          />

          <input
            name="email"
            type="email"
            placeholder="Email Address"
            value={formData.email}
            onChange={handleChange}
            required
          />

          <input
            name="apartment"
            placeholder="Apartment / Society Name"
            value={formData.apartment}
            onChange={handleChange}
            required
          />

          <input
            name="units"
            type="number"
            placeholder="Number of Units / Flats"
            value={formData.units}
            onChange={handleChange}
          />

          <textarea
            name="message"
            rows={4}
            placeholder="Tell us about your solar installation (capacity, rooftop size, etc.)"
            value={formData.message}
            onChange={handleChange}
            required
          />

          <button className="btn-primary" disabled={loading}>
            {loading ? "Submitting..." : "Request Solar Analysis"}
          </button>
        </form>
      </div>

    </div>

  </div>
);
};
