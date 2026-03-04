import React from "react";
import { NavLink, Link, useNavigate } from "react-router-dom";
import "../styles/navbar.css";

export default function Navbar({ isLoggedIn, setIsLoggedIn }) {
  const navigate = useNavigate();

  const logout = () => {
    localStorage.removeItem("user");
    setIsLoggedIn(false);
    navigate("/login");
  };

  return (
    <header className="site-header">
      <div className="nav-container">
        <Link to="/" className="brand">
          <img src="/assets/logo.png" alt="EcoCharge logo" className="logo" />
          <span className="brand-text">EcoCharge</span>
        </Link>

        <nav className="main-nav">
          <NavLink to="/" className="nav-link">Home</NavLink>
          <NavLink to="/about" className="nav-link">About</NavLink>
          <NavLink to="/contact" className="nav-link">Contact</NavLink>

          {!isLoggedIn && (
            <NavLink to="/login" className="nav-link">
              Login
            </NavLink>
          )}

          {isLoggedIn && (
            <>
              <NavLink to="/predict" className="nav-link">Predict</NavLink>
              <button className="nav-link logout-btn" onClick={logout}>
                Logout
              </button>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}
