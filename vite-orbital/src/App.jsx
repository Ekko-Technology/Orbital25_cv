import { useState } from 'react'
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css'
import Header from './components/header/header.jsx'
import Homebody from './components/home-body/home-body.jsx'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
    <Header/>
    <Homebody/>
    </>
  )
}

export default App
