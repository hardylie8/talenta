let mobilenet;
let mymodel;
const webcam = new Webcam(document.getElementById("wc"));
const dataset = new Dataset();
var totalLabel = 0;
sampleTotal = [];
categoryList = [];
let isPredicting = false;

async function loadMobilenet() {
    const mobilenet = await tf.loadLayersModel(
        "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json"
    );
    const layer = mobilenet.getLayer("conv_pw_13_relu");

    return tf.model({
        inputs: mobilenet.inputs,
        outputs: layer.output,
    });
}

async function predict() {
    mymodel = await tf.loadLayersModel(
        "https://ml-model.vercel.app/my_model.json"
    );
    console.log(mymodel);
    while (isPredicting) {
        const predictedClass = tf.tidy(() => {
            const img = webcam.capture();
            const activation = mobilenet.predict(img);
            const predictions = mymodel.predict(activation);
            return predictions.as1D().argMax();
        });

        const classId = (await predictedClass.data())[0];
        // var predictionText = "";
        // predictionText = categoryList[classId];
        document.getElementById("prediction").innerText = classId;

        predictedClass.dispose();
        await tf.nextFrame();
    }
}

function startPredicting() {
    isPredicting = true;
    predict();
}

function stopPredicting() {
    isPredicting = false;
    predict();
}

function doTraining() {
    train();
    swal({
        title: "Trainning Complete",
        text: "You can Start Predicting or Download the Model",
        icon: "success",
        button: "Close",
    });
}

async function init() {
    await webcam.setup();
    mobilenet = await loadMobilenet();
    tf.tidy(() => mobilenet.predict(webcam.capture()));
}

init();
