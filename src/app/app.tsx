import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, useNavigate } from 'react-router-dom';
import Comparison from '../pages/Home'

export function App() { 
    return ( 
        <BrowserRouter>
            <Routes>
                <Route
                    path="/home"
                    element={<Comparison />}
                />
            </Routes>
        </BrowserRouter>
    );
}

export default App;
