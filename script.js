import {MnistData} from './data.js';

console.log ('hello tf');
// stuff to initialize for drawing in canvas
// some of the code below was taken from a stackoverflow flag I
// not find anymore, and adapted to my needs.
// Thanks a ton to the original author!
var canvas;
var ctx; // for taking image from the canvas, context is initialized in init()

var prevX = 0;
var currX = 0;
var prevY = 0;
var currY = 0;
var paths = []; // recording paths of the mose in canvas
var paintFlag = false;
var color = 'black';
var lineWidth = 20;
var grayscaleImg = [];
var scaleStrokeWidth = true;
var hasTrainedModel = false;
var model = getModel ();

var clearBeforeDraw = false; // controls whether canvas will be cleared on next mousedown event. Set to true after digit recognition

async function showExamples (data) {
  // Create a container in the visor
  const surface = tfvis
    .visor ()
    .surface ({name: 'Input Data Examples', tab: 'Input Data'});

  // Get the examples
  const examples = data.nextTestBatch (20);
  const numExamples = examples.xs.shape[0];

  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy (() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice ([i, 0], [1, examples.xs.shape[1]])
        .reshape ([28, 28, 1]);
    });

    const canvas = document.createElement ('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels (imageTensor, canvas);
    surface.drawArea.appendChild (canvas);

    imageTensor.dispose ();
  }
}

async function showMyExamples (mydata) {
  // Create a container in the visor
  const surface = tfvis
    .visor ()
    .surface ({name: 'Input My Data Examples', tab: 'My Input Data'});

  // Get the examples

  console.log ('mydata');
  console.log (mydata);
  var reshaped = mydata.reshape ([28, 28, 1]);
  console.log ('reshaped');
  console.log (reshaped);

  const numExamples = 1;

  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    // const imageTensor = tf.tidy (() => {
    //   // Reshape the image to 28x28 px
    //   return reshaped
    //     .slice ([i, 0], [1, reshaped.shape[1]])
    //     .reshape ([28, 28, 1]);
    // });

    const canvas = document.createElement ('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';

    await tf.browser.toPixels (reshaped, canvas);

    surface.drawArea.appendChild (canvas);

    // reshaped.dispose ();
  }
}

// async function run () {
//   console.log ('trainbutton pushed, do all the stuff');
//   // get data
//   const data = new MnistData ();
//   //wait to load data
//   await data.load ();
//   // display some of the images
//   await showExamples (data);

//   // get a model
//   const model = getModel ();
//   // show the model
//   tfvis.show.modelSummary ({name: 'Model Architecture'}, model);

//   // wait until model is trained
//   await train (model, data);
//   // display stats of the model
//   await showAccuracy (model, data);
//   await showConfusion (model, data);
// }

// async function prep (model) {
//   console.log ('prepping the model');
//   // get data
//   const data = new MnistData ();
//   //wait to load data
//   await data.load ();
//   // display some of the images
//   await showExamples (data);

//   // get a model
//   // show the model
//   tfvis.show.modelSummary ({name: 'Model Architecture'}, model);

//   // wait until model is trained
//   if (
//     !hasTrainedModel ||
//     document.getElementById ('retrainModel').check == true
//   ) {
//     console.log ('training new model');
//     model = getModel ();
//     await train (model, data);
//     await showAccuracy (model, data);
//     await showConfusion (model, data);
//     hasTrainedModel = true;
//     // save model
//     await model.save ('localstorage://special');
//     console.log ('saved model');
//   } else {
//     // load model here
//     console.log ('loading model');
//     const model = await tf.loadLayersModel ('localstorage://special');
//   }
//   console.log ('Model is here');
//   console.log (model);
//   return model;
// }

// start ML
// define the structure of model
// no training yet
function getModel () {
  const model = tf.sequential ();

  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;

  // In the first layer of our convolutional neural network we have
  // to specify the input shape. Then we specify some parameters for
  // the convolution operation that takes place in this layer.
  model.add (
    tf.layers.conv2d ({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling',
    })
  );

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.
  model.add (tf.layers.maxPooling2d ({poolSize: [2, 2], strides: [2, 2]}));

  // Repeat another conv2d + maxPooling stack.
  // Note that we have more filters in the convolution.
  model.add (
    tf.layers.conv2d ({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling',
    })
  );
  model.add (tf.layers.maxPooling2d ({poolSize: [2, 2], strides: [1, 1]}));

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add (tf.layers.flatten ());

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_OUTPUT_CLASSES = 10;
  model.add (
    tf.layers.dense ({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax',
    })
  );

  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  const optimizer = tf.train.adam ();
  model.compile ({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

async function train (model, data) {
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = {
    name: 'Model Training',
    styles: {height: '1000px'},
  };
  const fitCallbacks = tfvis.show.fitCallbacks (container, metrics);

  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;

  const [trainXs, trainYs] = tf.tidy (() => {
    const d = data.nextTrainBatch (TRAIN_DATA_SIZE);
    return [d.xs.reshape ([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  const [testXs, testYs] = tf.tidy (() => {
    const d = data.nextTestBatch (TEST_DATA_SIZE);
    return [d.xs.reshape ([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  return model.fit (trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks,
  });
}

const classNames = [
  'Zero',
  'One',
  'Two',
  'Three',
  'Four',
  'Five',
  'Six',
  'Seven',
  'Eight',
  'Nine',
];

function doPrediction (model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch (testDataSize);

  const testxs = testData.xs.reshape ([
    testDataSize,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    1,
  ]);
  const labels = testData.labels.argMax (-1);
  const preds = model.predict (testxs).argMax (-1);

  testxs.dispose ();
  return [preds, labels];
}

async function doMyPrediction (model, data, testDataSize = 1) {
  // here we take data in as array of length 784
  // convert it to tensor
  // pass it to predict
  // return a prediction for 1 char
  console.log ('doMyPrediction');

  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  var tfdata = tf.reshape (data, [1, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  // t {isDisposedInternal: false, shape: Array(4), dtype: "float32", size: 784, strides: Array(3), …}

  console.log ('showMyexample');

  await showMyExamples (tfdata);

  // predict takes in tensor, spits out tensor
  // testxs {isDisposedInternal: false, shape: Array(4), dtype: "float32", size: 784, strides: Array(3), …}
  const preds = model.predict (tfdata);
  console.log (' real preds');
  console.log (preds.dataSync ());
  tfdata.dispose ();

  return preds.argMax (-1);
}

// functions that recognize need=============

// take canvas image and convert to grayscale. Mainly because my
// own functions operate easier on grayscale, but some stuff like
// resizing and translating is better done with the canvas functions
function imageDataToGrayscale (imgData) {
  var grayscaleImg = [];
  for (var y = 0; y < imgData.height; y++) {
    grayscaleImg[y] = [];
    for (var x = 0; x < imgData.width; x++) {
      var offset = y * 4 * imgData.width + 4 * x;
      var alpha = imgData.data[offset + 3];
      // weird: when painting with stroke, alpha == 0 means white;
      // alpha > 0 is a grayscale value; in that case I simply take the R value
      if (alpha == 0) {
        imgData.data[offset] = 255;
        imgData.data[offset + 1] = 255;
        imgData.data[offset + 2] = 255;
      }
      imgData.data[offset + 3] = 255;
      // simply take red channel value. Not correct, but works for
      // black or white images.
      grayscaleImg[y][x] =
        imgData.data[y * 4 * imgData.width + x * 4 + 0] / 255;
    }
  }

  // putting it onto the canvas
  // var gc = document.getElementById ('grey');
  // var gctx = gc.getContext ('2d');
  // var gimgData = gctx.createImageData (280, 280);
  // var gi;
  // var grow;
  // var gcol;
  // // gimgData.data.length = 280*280*4
  // for (gi = 0; gi < gimgData.data.length; gi += 4) {
  //   grow = Math.floor (gi / 4 / 280);
  //   gcol = Math.floor (gi / 4) % 280;
  //   // console.log (grow, gcol);
  //   // console.log (grayscaleImg[grow][gcol]);
  //   gimgData.data[gi + 0] = grayscaleImg[grow][gcol] * 255;
  //   gimgData.data[gi + 1] = 255;
  //   gimgData.data[gi + 2] = 255;
  //   gimgData.data[gi + 3] = 255;
  // }
  // gctx.putImageData (gimgData, 10, 10);

  // putDataToCanvas ('grey', grayscaleImg, 280, 280);

  return grayscaleImg;
}

function putDataToCanvas (id, imgdata, w, h) {
  console.log ('drawing on', id);
  var gc = document.getElementById (id);
  var gctx = gc.getContext ('2d');
  var gimgData = gctx.createImageData (w, h);
  var gi;
  var grow;
  var gcol;
  // gimgData.data.length = 280*280*4
  for (gi = 0; gi < gimgData.data.length; gi += 4) {
    grow = Math.floor (gi / 4 / w);
    gcol = Math.floor (gi / 4) % w;
    // console.log (grow, gcol);
    // console.log (grayscaleImg[grow][gcol]);
    gimgData.data[gi + 0] = imgdata[grow][gcol] * 255;
    gimgData.data[gi + 1] = 255;
    gimgData.data[gi + 2] = 255;
    gimgData.data[gi + 3] = 255;
  }
  gctx.putImageData (gimgData, 0, 0);
}

function putDataToCanvas2 (id, oldimgdata, w, h) {
  var imgdata = [];
  var newArr = [];
  while (oldimgdata.length)
    imgdata.push (oldimgdata.splice (0, w));
  console.log ('drawing on', id);
  var gc = document.getElementById (id);
  var gctx = gc.getContext ('2d');
  var gimgData = gctx.createImageData (w, h);
  var gi;
  var grow;
  var gcol;
  // gimgData.data.length = 280*280*4
  for (gi = 0; gi < gimgData.data.length; gi += 4) {
    grow = Math.floor (gi / 4 / w);
    gcol = Math.floor (gi / 4) % w;
    // console.log (grow, gcol);
    // console.log (grayscaleImg[grow][gcol]);
    gimgData.data[gi + 0] = imgdata[grow][gcol] * 255;
    gimgData.data[gi + 1] = 255;
    gimgData.data[gi + 2] = 255;
    gimgData.data[gi + 3] = 255;
  }
  gctx.putImageData (gimgData, 0, 0);
}

// given grayscale image, find bounding rectangle of digit defined
// by above-threshold surrounding
function getBoundingRectangle (img, threshold) {
  var rows = img.length;
  var columns = img[0].length;
  var minX = columns;
  var minY = rows;
  var maxX = -1;
  var maxY = -1;
  for (var y = 0; y < rows; y++) {
    for (var x = 0; x < columns; x++) {
      if (img[y][x] < threshold) {
        if (minX > x) minX = x;
        if (maxX < x) maxX = x;
        if (minY > y) minY = y;
        if (maxY < y) maxY = y;
      }
    }
  }
  return {minY: minY, minX: minX, maxY: maxY, maxX: maxX};
}

// computes center of mass of digit, for centering
// note 1 stands for black (0 white) so we have to invert.
function centerImage (img) {
  var meanX = 0;
  var meanY = 0;
  var rows = img.length;
  var columns = img[0].length;
  var sumPixels = 0;
  for (var y = 0; y < rows; y++) {
    for (var x = 0; x < columns; x++) {
      var pixel = 1 - img[y][x];
      sumPixels += pixel;
      meanY += y * pixel;
      meanX += x * pixel;
    }
  }
  meanX /= sumPixels;
  meanY /= sumPixels;

  var dY = Math.round (rows / 2 - meanY);
  var dX = Math.round (columns / 2 - meanX);
  return {transX: dX, transY: dY};
}

// start of recognize==========================
async function recognize () {
  var t1 = new Date (); // for calculating prediction time

  // taking data from ctx
  // convert RGBA image to a grayscale array, then compute bounding rectangle and center of mass
  var imgData = ctx.getImageData (0, 0, 280, 280);
  console.log ('imgData');
  console.log (imgData);
  grayscaleImg = imageDataToGrayscale (imgData); // a list of list
  console.log ('grayscaleImg');
  console.log (grayscaleImg);
  // returns a dict of vertex of rectangle
  var boundingRectangle = getBoundingRectangle (grayscaleImg, 0.01);
  //return dict with center x, y of image
  var trans = centerImage (grayscaleImg); // [dX, dY] to center of mass

  // copy image to hidden canvas, translate to center-of-mass, then
  // scale to fit into a 200x200 box (see MNIST calibration notes on
  // Yann LeCun's website)
  var canvasCopy = document.createElement ('canvas');
  canvasCopy.width = imgData.width;
  canvasCopy.height = imgData.height;
  var copyCtx = canvasCopy.getContext ('2d');
  var brW = boundingRectangle.maxX + 1 - boundingRectangle.minX;
  var brH = boundingRectangle.maxY + 1 - boundingRectangle.minY;
  var scaling = 190 / (brW > brH ? brW : brH);
  // scale
  copyCtx.translate (canvas.width / 2, canvas.height / 2);
  copyCtx.scale (scaling, scaling);
  copyCtx.translate (-canvas.width / 2, -canvas.height / 2);
  // translate to center of mass
  copyCtx.translate (trans.transX, trans.transY);

  // if (document.getElementById ('scaleStrokeWidth').checked == true) {
  // redraw the image with a scaled lineWidth first.
  // not this is a bit buggy; the bounding box we computed above (which contributed to "scaling") is not valid anymore because
  // the line width has changed. This is mostly a problem for extreme cases (very small digits) where the rescaled digit will
  // be smaller than the bounding box. I could change this but it'd screw up the code.
  // for (var p = 0; p < paths.length; p++) {
  //   for (var i = 0; i < paths[p][0].length - 1; i++) {
  //     var x1 = paths[p][0][i];
  //     var y1 = paths[p][1][i];
  //     var x2 = paths[p][0][i + 1];
  //     var y2 = paths[p][1][i + 1];
  //     draw (copyCtx, color, lineWidth / scaling, x1, y1, x2, y2);
  //   }
  // }
  // } else {
  // default take image from original canvas
  copyCtx.drawImage (ctx.canvas, 0, 0);
  // }

  // now bin image into 10x10 blocks (giving a 28x28 image)
  imgData = copyCtx.getImageData (0, 0, 280, 280);
  grayscaleImg = imageDataToGrayscale (imgData);
  console.log ('imgData');
  console.log (imgData.data);
  console.log ('grayscaleImg');
  console.log (grayscaleImg);
  var nnInput = new Array (784);
  for (var y = 0; y < 28; y++) {
    for (var x = 0; x < 28; x++) {
      var mean = 0;
      for (var v = 0; v < 10; v++) {
        for (var h = 0; h < 10; h++) {
          mean += grayscaleImg[y * 10 + v][x * 10 + h];
        }
      }
      mean = 1 - mean / 100; // average and invert
      // mean = mean / 100; // try not inverting
      // console.log (mean);
      // nnInput[x * 28 + y] = (mean - 0.5) / 0.5;
      nnInput[y * 28 + x] = mean; // let input be [0,1]
    }
  }
  console.log ('nnInput');
  console.log (nnInput);

  // putDataToCanvas2 ('nncanvas', nnInput, 28, 28);

  // for visualization/debugging: paint the input to the neural net.
  if (document.getElementById ('preprocessing').checked == true) {
    ctx.clearRect (0, 0, canvas.width, canvas.height);
    ctx.drawImage (copyCtx.canvas, 0, 0);
    for (var y = 0; y < 28; y++) {
      for (var x = 0; x < 28; x++) {
        var block = ctx.getImageData (x * 10, y * 10, 10, 10);
        var newVal = 255 * (0.5 - nnInput[y * 28 + x] / 2);
        for (var i = 0; i < 4 * 10 * 10; i += 4) {
          block.data[i] = newVal;
          block.data[i + 1] = newVal;
          block.data[i + 2] = newVal;
          block.data[i + 3] = 255;
        }
        ctx.putImageData (block, x * 10, y * 10);
      }
    }
  }

  var onCanvas = ctx.getImageData (0, 0, 280, 280);
  console.log ('onCanvas');
  console.log (onCanvas);

  console.log ('nnInput');
  console.log (nnInput); // array of length 784

  if (
    !hasTrainedModel ||
    document.getElementById ('retrainModel').check == true
  ) {
    console.log ('training new');
    const mnistdata = new MnistData ();
    //wait to load data
    await mnistdata.load ();
    // display some of the images
    await showExamples (mnistdata);

    tfvis.show.modelSummary ({name: 'Model Architecture'}, model);

    // wait until model is trained
    await train (model, mnistdata);
    // display stats of the model
    await showAccuracy (model, mnistdata);
    await showConfusion (model, mnistdata);
    hasTrainedModel = true;
  }

  // predict
  var prediction = await doMyPrediction (model, nnInput);

  console.log ('prediction: ');
  console.log (prediction);
  document.getElementById ('prediction').innerHTML = prediction;
  console.log ('prediction:');
  console.log (prediction);
  clearBeforeDraw = true;
  var dt = new Date () - t1;
  console.log ('recognize time: ' + dt + 'ms');
}

// ================end of recognize

async function showAccuracy (model, data) {
  const [preds, labels] = doPrediction (model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy (labels, preds);
  const container = {name: 'Accuracy', tab: 'Evaluation'};
  tfvis.show.perClassAccuracy (container, classAccuracy, classNames);

  labels.dispose ();
}

async function showConfusion (model, data) {
  const [preds, labels] = doPrediction (model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix (labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  tfvis.render.confusionMatrix (
    container,
    {values: confusionMatrix},
    classNames
  );

  labels.dispose ();
}

// stuff for drawing digit in canvas from myselph.de

// initialize the canvas
// listens to the mouse moves
function init () {
  canvas = document.getElementById ('can');
  ctx = canvas.getContext ('2d');

  //pure moving the mouse around
  canvas.addEventListener (
    'mousemove',
    function (e) {
      findxy ('move', e);
    },
    false
  );
  //pressing down with mouse
  canvas.addEventListener (
    'mousedown',
    function (e) {
      findxy ('down', e);
    },
    false
  );
  //releasing mouse
  canvas.addEventListener (
    'mouseup',
    function (e) {
      findxy ('up', e);
    },
    false
  );
  //movement outside of frame
  canvas.addEventListener (
    'mouseout',
    function (e) {
      findxy ('out', e);
    },
    false
  );
}

// draw lines when we move the mouse, draw is called every time we move the mouse
// draws a line from (x1, y1) to (x2, y2) with nice rounded caps
function draw (ctx, color, lineWidth, x1, y1, x2, y2) {
  ctx.beginPath ();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.moveTo (x1, y1);
  ctx.lineTo (x2, y2);
  ctx.stroke ();
  ctx.closePath ();
}

// store the path of mouse into path
function findxy (res, e) {
  if (res == 'down') {
    if (clearBeforeDraw == true) {
      //make sure we have a clean canvas
      ctx.clearRect (0, 0, canvas.width, canvas.height);
      document.getElementById ('prediction').innerHTML = '';
      paths = [];
      clearBeforeDraw = false;
      //dont clean for the next stroke
    }

    if (e.pageX != undefined && e.pageY != undefined) {
      currX = e.pageX - canvas.offsetLeft;
      currY = e.pageY - canvas.offsetTop;
    } else {
      currX =
        e.clientX +
        document.body.scrollLeft +
        document.documentElement.scrollLeft -
        canvas.offsetLeft;
      currY =
        e.clientY +
        document.body.scrollTop +
        document.documentElement.scrollTop -
        canvas.offsetTop;
    }
    //draw a circle
    ctx.beginPath ();
    ctx.lineWidth = 1;
    ctx.arc (currX, currY, lineWidth / 2, 0, 2 * Math.PI);
    ctx.stroke ();
    ctx.closePath ();
    ctx.fill ();

    paths.push ([[currX], [currY]]);
    paintFlag = true;
  }
  if (res == 'up' || res == 'out') {
    paintFlag = false;
    //console.log(paths);
  }

  if (res == 'move') {
    if (paintFlag) {
      // draw a line to previous point
      prevX = currX;
      prevY = currY;
      if (e.pageX != undefined && e.pageY != undefined) {
        currX = e.pageX - canvas.offsetLeft;
        currY = e.pageY - canvas.offsetTop;
      } else {
        currX =
          e.clientX +
          document.body.scrollLeft +
          document.documentElement.scrollLeft -
          canvas.offsetLeft;
        currY =
          e.clientY +
          document.body.scrollTop +
          document.documentElement.scrollTop -
          canvas.offsetTop;
      }
      var currPath = paths[paths.length - 1];
      currPath[0].push (currX);
      currPath[1].push (currY);
      paths[paths.length - 1] = currPath;
      draw (ctx, color, lineWidth, prevX, prevY, currX, currY);
      // console.log (currPath);
    }
  }
}

init ();
document.getElementById ('predict').addEventListener ('click', recognize);
