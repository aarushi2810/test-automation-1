import React, { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import Header from "./components/Header";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";

import Home from "./pages/Home";
import Predict from "./pages/Predict";
import About from "./pages/About";
import Contact from "./pages/Contact";
import Register from "./pages/Register";
import Login from "./pages/Login";

import "./App.css";

function App() {
  const [prediction, setPrediction] = useState("");
  const [isLoggedIn, setIsLoggedIn] = useState(
    !!localStorage.getItem("user")
  );

  useEffect(() => {
    const last = localStorage.getItem("lastPrediction");
    if (last) setPrediction(parseFloat(last).toFixed(2));

    const handleStorage = () => {
      setIsLoggedIn(!!localStorage.getItem("user"));
    };

    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, []);


  return (
    <BrowserRouter>
      <div className="app">
        <Header />
        <Navbar isLoggedIn={isLoggedIn} setIsLoggedIn={setIsLoggedIn} />


        <main className="main-content">
          <div className="container">
            <Routes>
              <Route path="/" element={<Home />} />

              {/* 🔐 Protected Route */}
              <Route
                path="/predict"
                element={
                  isLoggedIn ? (
                    <Predict
                      setPrediction={setPrediction}
                      prediction={prediction}
                    />
                  ) : (
                    <Navigate to="/login" />
                  )
                }
              />

              <Route path="/about" element={<About />} />
              <Route path="/contact" element={<Contact />} />
              <Route path="/register" element={<Register />} />
              <Route path="/login" element={<Login setIsLoggedIn={setIsLoggedIn} />}/>
/>


              <Route path="*" element={<Home />} />
            </Routes>
          </div>
        </main>

        <Footer />
      </div>
    </BrowserRouter>
  );
}

export default App;
