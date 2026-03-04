import React, { useState } from "react"; 
import "../styles/page.css"; 
import { useNavigate } from "react-router-dom";
const Register = () => { const navigate = useNavigate(); const [formData, setFormData] = useState({ name: "", email: "", password: "", confirmPassword: "" }); const [error, setError] = useState(""); const [success, setSuccess] = useState(""); const [isRegistered, setIsRegistered] = useState(false); const handleChange = (e) => { setFormData({ ...formData, [e.target.name]: e.target.value }); }; 
const validate = () => {
  const { name, email, password, confirmPassword } = formData;

  if (!name || !email || !password || !confirmPassword)
    return "All fields are required";

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email))
    return "Invalid email format";

  // Password Rules
  if (password.length < 8 || password.length > 15)
    return "Password must be between 8 and 15 characters";

  if (!/[0-9]/.test(password))
    return "Password must contain at least one digit";

  if (!/[A-Z]/.test(password))
    return "Password must contain at least one uppercase letter";

  if (!/[a-z]/.test(password))
    return "Password must contain at least one lowercase letter";

  if (!/[!@#$%&*()\-\+=^\.]/.test(password))
    return "Password must contain at least one special character (!@#$%&*()-+=^.)";

  if (/\s/.test(password))
    return "Password must not contain spaces";

  if (password !== confirmPassword)
    return "Passwords do not match";

  return "";
};
const handleSubmit = async (e) => { e.preventDefault(); setError(""); setSuccess(""); const validationError = validate(); if (validationError) { setError(validationError); return; } try { const res = await fetch("http://localhost:5001/register", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: formData.name, email: formData.email, password: formData.password }) }); const data = await res.json(); if (!res.ok) throw new Error(data.error); setIsRegistered(true); setTimeout(() => { navigate("/login"); }, 2000); } catch (err) { setError(err.message); } };
return (
  <div className="page-auth">
    <div className="auth-card">
      <h2>Create Account</h2>

      {isRegistered ? (
        <div className="success-box">
          <p className="success-message">
            ✅ Registration completed successfully!
          </p>
          <span className="success-subtext">
            Redirecting to login...
          </span>
        </div>
      ) : (
        <>
          {error && <p id="errorMsg" className="error-message">{error}</p>}

          <form onSubmit={handleSubmit}>
            <input
              type="text"
              name="name"
              placeholder="Full Name"
              value={formData.name}
              onChange={handleChange}
              required
            />

            <input
              type="email"
              name="email"
              placeholder="Email Address"
              value={formData.email}
              onChange={handleChange}
              required
            />

            <input
              type="password"
              id="password"
              name="password"
              placeholder="Password"
              value={formData.password}
              onChange={handleChange}
              required
            />

            <input
              type="password"
              id="confirm password"
              name="confirmPassword"
              placeholder="Confirm Password"
              value={formData.confirmPassword}
              onChange={handleChange}
              required
            />

            <button id="registerBtn" type="submit" className="auth-btn">
              Register
            </button>
          </form>

          <div className="auth-switch">
            <p>Already have an account?</p>
            <span
              onClick={() => navigate("/login")}
              className="auth-link"
            >
              Login
            </span>
          </div>
        </>
      )}
    </div>
  </div>
);
};
export default Register;