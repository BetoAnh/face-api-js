const container = document.querySelector("#container");
const fileInput = document.querySelector("#file-input");

async function saveTrainingData(faceDescriptors) {
  try {
    const json = JSON.stringify(faceDescriptors.map(fd => ({
      label: fd.label,
      descriptors: fd.descriptors.map(descriptor => Array.from(descriptor))
    })));
    localStorage.setItem('trainingData', json);
  } catch (error) {
    console.error('Error saving training data:', error);
  }
}

async function loadTrainingData() {
  try {
    const savedData = localStorage.getItem('trainingData');
    if (savedData) {
      const data = JSON.parse(savedData);
      return data.map(fd => new faceapi.LabeledFaceDescriptors(
        fd.label,
        fd.descriptors.map(descriptor => new Float32Array(descriptor))
      ));
    }

    const labels = [
      "Andrew Garfield",
      "Chris Hemsworth",
      "Johnny Depp",
      "Jack",
      "Justin Bieber",
      "Keanu Reeves",
      "Kim Moo Yul",
      "Ma Dong-seok",
      "Robert Downey Jr",
      "Sơn Tùng M-TP",
      "Tăng Duy Tân",
      "Tom Cruise",
      "Will Smith",
    ];

    const faceDescriptors = [];
    for (const label of labels) {
      const descriptors = [];
      for (let i = 1; i <= 20; i++) {
        const image = await faceapi.fetchImage(`/data/${label}/${i}.jpg`);
        const detection = await faceapi
          .detectSingleFace(image)
          .withFaceLandmarks()
          .withFaceDescriptor();
        if (detection) descriptors.push(detection.descriptor);
      }
      faceDescriptors.push(
        new faceapi.LabeledFaceDescriptors(label, descriptors)
      );

      Toastify({
        text: `Training xong data của ${label}!`,
      }).showToast();
    }

    await saveTrainingData(faceDescriptors);

    return faceDescriptors;
  } catch (error) {
    console.error('Error loading training data:', error);
    return [];
  }
}

let faceMatcher;
async function init() {
  try {
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
      faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
      faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
    ]);

    Toastify({
      text: "Tải xong model nhận diện!",
    }).showToast();

    const trainingData = await loadTrainingData();
    faceMatcher = new faceapi.FaceMatcher(trainingData, 0.6);

    console.log(faceMatcher);
    document.querySelector("#loading").remove();
  } catch (error) {
    console.error('Error initializing models:', error);
  }
}

init();

fileInput.addEventListener("change", async () => {
  try {
    const files = fileInput.files;
    if (files.length === 0) return;

    const image = await faceapi.bufferToImage(files[0]);
    const canvas = faceapi.createCanvasFromMedia(image);

    container.innerHTML = "";
    container.append(image);
    container.append(canvas);

    const size = {
      width: image.width,
      height: image.height,
    };

    faceapi.matchDimensions(canvas, size);

    const detections = await faceapi
      .detectAllFaces(image)
      .withFaceLandmarks()
      .withFaceDescriptors();
    const resizedDetections = faceapi.resizeResults(detections, size);

    for (const detection of resizedDetections) {
      const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
      const label = `${bestMatch.label} (${Math.round(
        bestMatch.distance * 100
      )}%)`;
      const drawBox = new faceapi.draw.DrawBox(detection.detection.box, {
        label: label,
      });
      drawBox.draw(canvas);
    }
  } catch (error) {
    console.error('Error during face detection:', error);
    Toastify({
      text: "Đã xảy ra lỗi trong việc nhận diện khuôn mặt.",
    }).showToast();
  }
});

// function getLocalStorageSize() {
//   let total = 0;
//   for (let i = 0; i < localStorage.length; i++) {
//     const key = localStorage.key(i);
//     const value = localStorage.getItem(key);
//     total += key.length + value.length;
//   }
//   return total;
// }
// const sizeInBytes = getLocalStorageSize();
// console.log(`Dung lượng sử dụng: ${sizeInBytes} bytes`);
// const sizeInKB = sizeInBytes / 1024;
// console.log(`Dung lượng sử dụng: ${sizeInKB.toFixed(2)} KB`);