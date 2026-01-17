import axios from "axios";
import { useRef, useState } from "react";
import { RotatingLines } from "react-loader-spinner";
import "./App.css";
import { ProbabilityBar } from "./components/ProbabilityBar";

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState<boolean>(false);
  const [prevCoords, setPrevCoords] = useState<number[] | null>(null);
  const [numberGuess, setNumberGuess] = useState<number[]>([]);
  const [isWakingUp, setIsWakingUp] = useState<boolean>(false);

  const draw = (e: React.MouseEvent) => {
    if (!isDrawing) return;

    const c = canvasRef.current;
    if (!c) return;

    const ctx = c.getContext("2d");
    if (!ctx) return;

    ctx.lineWidth = 10;
    ctx.shadowBlur = 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.shadowColor = "black";
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    const x = e.nativeEvent.offsetX;
    const y = e.nativeEvent.offsetY;

    ctx.beginPath();

    if (prevCoords) {
      ctx.moveTo(prevCoords[0], prevCoords[1]);
      ctx.lineTo(x, y);
      ctx.stroke();
    }
    setPrevCoords([x, y]);
  };

  const clearCanvas = () => {
    if (isWakingUp) return alert("Wait for the server to respond");

    const c = canvasRef.current;
    if (!c) return;

    const ctx = c.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, c.width, c.height);
    setNumberGuess([]);
    setPrevCoords(null);
  };

  const recognizeNumber = async () => {
    if (isWakingUp) return alert("Wait for the server to respond");

    const c = canvasRef.current;
    if (!c) return;

    const ctx = c.getContext("2d");
    if (!ctx) return;

    setIsWakingUp(true);

    const imageData = ctx.getImageData(0, 0, c.width, c.height);
    const pixels = Array.from(imageData.data);

    const filtered_pixels: number[] = [];
    for (let i = 3; i < pixels.length; i += 4) {
      filtered_pixels.push(pixels[i]);
    }

    try {
      const response = await axios.post(
        "https://mnist-web-recognizer-1.onrender.com",
        {
          data: filtered_pixels,
        }
      );
      setNumberGuess(response.data[0]);
    } catch (error) {
      console.log(error);
    } finally {
      setIsWakingUp(false);
    }
  };

  return (
    <>
      <h1 className="title">Neural Network Based Number Recognizer</h1>
      <div className={"mainContainer"}>
        <div className={"canvasContainer"}>
          <p className={"explanationHeader"}>
            Draw a number and the neural network will guess what it is
          </p>
          <canvas
            ref={canvasRef}
            className={"digitCanvas"}
            onMouseDown={() => setIsDrawing(true)}
            onMouseUp={() => {
              setIsDrawing(false);
              setPrevCoords(null);
            }}
            onMouseLeave={() => {
              setIsDrawing(false);
              setPrevCoords(null);
            }}
            onMouseMove={(e) => draw(e)}
            width={400}
            height={400}
          ></canvas>
          <div className={"buttonContainer"}>
            <button onClick={clearCanvas}>Clear</button>
            <button onClick={recognizeNumber}>
              {isWakingUp ? <RotatingLines height="20" color="blue" /> : "Run"}
            </button>
          </div>
        </div>
        <div className="probabilityContainer">
          <ProbabilityBar numberGuess={numberGuess}></ProbabilityBar>
        </div>
      </div>
    </>
  );
}

export default App;
