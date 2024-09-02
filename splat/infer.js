let mobilenet;
let model;
// const webcam = new Webcam(document.getElementById('wc'));
// const dataset = new RPSDataset();
var rockSamples=0, paperSamples=0, scissorsSamples=0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadGraphModel('http://10.10.22.222:8000/outtfjs/model.json');
//   console.log('==');
//   const layer = mobilenet.getLayer('conv_pw_13_relu');
//   return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
    return mobilenet;
}

async function init(){
	mobilenet = await loadMobilenet();
    const zeros = tf.randomNormal([1, 418]);
	tf.tidy(() => mobilenet.predict(zeros));
    let i;
    for(i=0;i<100;i++) {
        const zeros = tf.randomNormal([1, 418]);
        tf.tidy(() => mobilenet.predict(zeros));
        console.log(i);
        document.getElementById("prediction").innerText = i;
    }
}

async function runModel() {
    // try {
    //     model = await tf.loadGraphModel('http://10.10.22.222:8000/outtfjs/model.json');
    //     const inputTensor = tf.randomNormal([1, 418]);
    //     const predictions = await model.execute(inputTensor);
    //     console.log(predictions);
    // } catch (err) {
    //     console.error('Error during model execution:', err);
    // }
    model = await tf.loadGraphModel('http://10.10.22.222:8000/outtfjs/model.json');
    const inputTensor = tf.randomNormal([418]);
    const res = await model.executeAsync(inputTensor);
    const rotations = res[0];
    const xyz = res[1];
    const scales = res[2];
    const opacity = res[3];
    const shs = res[4];

    // console.log(xyz.print());
    // console.log(scales.print());

    // redictions = model.executeAsync(inputTensor, ['xyz', 'rotations']).then(predictions=> { 
    //     const data = predictions.dataSync() // you can also use arraySync or their equivalents async methods
    //     console.log('Predictions: ', data);
    //   })
}
runModel();

// init();