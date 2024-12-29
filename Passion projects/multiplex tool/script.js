document.getElementById('fileInput').addEventListener('change', handleFileUpload);

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file || file.type !== "text/csv") {
        alert("Please upload a valid CSV file.");
        return;
    }

    const progressBar = document.getElementById('progressBar');
    const message = document.getElementById('message');
    progressBar.style.display = "block";
    message.textContent = "Uploading...";

    const reader = new FileReader();
    reader.onprogress = function (e) {
        if (e.lengthComputable) {
            progressBar.value = (e.loaded / e.total) * 100;
        }
    };

    reader.onload = function () {
        progressBar.value = 100;
        message.textContent = "File uploaded successfully!";
        document.getElementById('questions').style.display = "block";

        // Store file data for later processing
        window.csvData = reader.result;
    };

    reader.readAsText(file);
}

document.getElementById('networkForm').addEventListener('change', function (event) {
    if (event.target.name === "isNetwork" && event.target.value === "yes") {
        document.getElementById('networkQuestions').style.display = "block";
    } else if (event.target.name === "isNetwork" && event.target.value === "no") {
        document.getElementById('networkQuestions').style.display = "none";
    }

    if (event.target.name === "isWeighted" && event.target.value === "yes") {
        document.getElementById('weightQuestion').style.display = "block";
    } else if (event.target.name === "isWeighted" && event.target.value === "no") {
        document.getElementById('weightQuestion').style.display = "none";
    }
});

document.getElementById('processButton').addEventListener('click', function () {
    const isNetwork = document.querySelector('input[name="isNetwork"]:checked').value === "yes";
    if (!isNetwork) {
        alert("This tool only processes network files.");
        return;
    }

    const isDirected = document.querySelector('input[name="isDirected"]:checked').value === "yes";
    const sourceColumn = document.getElementById('sourceColumn').value;
    const targetColumn = document.getElementById('targetColumn').value;

    const isWeighted = document.querySelector('input[name="isWeighted"]:checked').value === "yes";
    const weightColumn = isWeighted ? document.getElementById('weightColumn').value : null;

    const data = parseCSV(window.csvData);
    const filteredData = filterNetworkData(data, sourceColumn, targetColumn, weightColumn, isDirected);

    previewData(filteredData, isWeighted);
});

function parseCSV(csvData) {
    return csvData.split("\n").map(row => row.split(","));
}

function filterNetworkData(data, source, target, weight, directed) {
    const headers = data[0];
    const sourceIndex = headers.indexOf(source);
    const targetIndex = headers.indexOf(target);
    const weightIndex = weight ? headers.indexOf(weight) : -1;

    if (sourceIndex === -1 || targetIndex === -1 || (weight && weightIndex === -1)) {
        alert("Column names do not match the uploaded file.");
        return [];
    }

    const filteredData = data.slice(1).map(row => {
        const sourceVal = row[sourceIndex];
        const targetVal = row[targetIndex];
        const weightVal = weight ? row[weightIndex] : null;

        return directed ? 
            [sourceVal, targetVal, weightVal] : 
            [[sourceVal, targetVal, weightVal], [targetVal, sourceVal, weightVal]];
    }).flat();

    return weight ? [["source", "target", "weight"], ...filteredData] : [["source", "target"], ...filteredData];
}

function previewData(filteredData, isWeighted) {
    const previewDiv = document.getElementById('preview');
    const table = document.getElementById('dataPreview');
    table.innerHTML = ""; // Clear previous data

    filteredData.forEach((row, index) => {
        const tr = document.createElement('tr');
        row.forEach(cell => {
            const td = document.createElement(index === 0 ? 'th' : 'td');
            td.textContent = cell;
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });

    previewDiv.style.display = "block";
}
