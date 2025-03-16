
const canvas = document.querySelector("canvas"),
ctx = canvas.getContext("2d",{ willReadFrequently: true });
clearCanvas = document.querySelector(".clearButtomCnavas");
const sendImage = document.getElementById('imagenn');

const canvasOffsetX = canvas.offsetLeft
const canvasOffsetY = canvas.offsetTop

let isDrawning = false, brushWidth = 10;

window.addEventListener("load", () => {
  // setting canvas width and heigt
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
});

const StarDraw = () => {
  isDrawning = true;
  ctx.beginPath(); //creating a new path to draw, withut this the lines came across
  ctx.lineWidth = brushWidth; //passing the width of brush
}

const drawning = (e) => {
  if(!isDrawning) return; //if isDrawing is false return from here
  ctx.lineTo(e.offsetX, e.offsetY); // Creating the line
  ctx.stroke(); // puting color in the line
}

clearCanvas.addEventListener("click", () =>{
  ctx.fillStyle = "#fff";
  ctx.clearRect(0,0,canvas.width,canvas.height);
})

sendImage.addEventListener('click', () => { 
  
  const blackAndWhiteArray = [];
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;

  // Loop through each pixel
  for (let i = 3; i < data.length; i += 4) {
    const t = data[i]/255;     // Transparent
    // Apply threshold to determine black (0) or white (1)
    blackAndWhiteArray.push(t);
  }
  
  const image = blackAndWhiteArray;
  const url = '/NeuralNetwork';

fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({data: image})
})
.then(response => response.json())
.then(result => {
    console.log('Prediction:', result.prediction);
    console.log('Accuracy:', result.accuracy);

    // Display result on the webpage
    document.getElementById('result').innerHTML = 
      `The number you drew is ${result.prediction}, <br> I can say with ${result.accuracy}% certainty !!`;
})
.catch(error => {
  console.error('Error:', error);
});

});

canvas.addEventListener("mousemove", drawning); //Waiting for the movement of the mouse, them calling the function "Drawning"
canvas.addEventListener("mousedown", StarDraw); //Waiting for the moude down of the mouse, them calling the function "StartDrawning"
canvas.addEventListener("mouseup", () => isDrawning = false); // Waiting for the moude up of the mouse, them calling an internal function that disable the IsDrawning variable