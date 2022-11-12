let mobilenet;
let model;
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

async function train() {
    dataset.ys = null;
    dataset.encodeLabels(5);

    model = tf.sequential({
        layers: [
            tf.layers.flatten({
                inputShape: mobilenet.outputs[0].shape.slice(1),
            }),
            tf.layers.dense({
                units: 100,
                activation: "relu",
            }),
            tf.layers.dense({
                units: 5,
                activation: "softmax",
            }),
        ],
    });

    model.compile({
        loss: "categoricalCrossentropy",
        optimizer: tf.train.adam(0.0001),
    });

    model.summary();
    let loss = 0;
    model.fit(dataset.xs, dataset.ys, {
        epochs: 10,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                loss = logs.loss.toFixed(5);
                console.log("LOSS: " + loss);
            },
        },
    });
}

function handleButton(elem) {
    label = parseInt(elem.id);
    ++sampleTotal[elem.id];
    document.getElementById(categoryList[label]).innerText = sampleTotal[label];
    const img = webcam.capture();
    dataset.addExample(mobilenet.predict(img), label);
}

async function predict() {
    // const mymodel = await tf.loadLayersModel(
    //     "https://ml-model.vercel.app/my_model.json"
    // );
    // console.log(mymodel);
    while (isPredicting) {
        const predictedClass = tf.tidy(() => {
            const img = webcam.capture();
            const activation = mobilenet.predict(img);
            const predictions = model.predict(activation);
            return predictions.as1D().argMax();
        });

        const classId = (await predictedClass.data())[0];
        var predictionText = "";
        predictionText = categoryList[classId];
        document.getElementById("prediction").innerText = predictionText;

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

function addNewCategory() {
    let inputCategory = document.getElementById("categoryInput").value;
    if (inputCategory !== "") {
        categoryList[totalLabel] = inputCategory;
        sampleTotal[totalLabel] = 0;
        let el = document.createElement("tr");
        el.innerHTML = `<td class="w-50" >${inputCategory}</td>
                          <td class="text-center" id="${categoryList[totalLabel]}">0</td>
                          <td class="text-right font-weight-medium">
                          <button type="button" class="btn btn-rounded btn-primary pl-2 btn-icon"><i class="fa fa-plus"></i></button>
                           <button type="button" class="btn btn-rounded btn-success pl-2 btn-icon"><i class="fa fa-pencil"></i></button>
                           <button type="button" class="btn btn-rounded btn-danger pl-2 btn-icon"><i class="fa fa-trash"></i></button>

                           
                          </td>`;
        const box = document.getElementById("categoryHolder");
        box.appendChild(el);
        document.getElementById("categoryInput").value = "";
        totalLabel++;
        if (totalLabel > 0) {
            document.getElementById("train").disabled = false;
        }
    }
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

function saveModel() {
    model.save("downloads://my_model");
}

async function init() {
    await webcam.setup();
    mobilenet = await loadMobilenet();
    tf.tidy(() => mobilenet.predict(webcam.capture()));
}

init();
