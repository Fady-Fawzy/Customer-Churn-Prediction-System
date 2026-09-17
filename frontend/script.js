const welcomeSection = document.querySelector(".welcome-section");

const singleSection = document.getElementById("single-section");
const singleButton = document.getElementById("single-option");
const backFromSingle = document.getElementById("back-from-single");

const predictButton = document.getElementById("predict-button");

const resultCard = document.getElementById("result-card");
const resultPrediction = document.getElementById("result-prediction");
const resultProbability = document.getElementById("result-probability");
const resultRisk = document.getElementById("result-risk");


// =========================
// Batch CSV Elements
// =========================

const batchButton = document.getElementById("batch-option");
const batchSection = document.getElementById("batch-section");
const backFromBatch = document.getElementById("back-from-batch");

const csvFileInput = document.getElementById("csv-file");

const fileInfo = document.getElementById("file-info");
const fileName = document.getElementById("file-name");

const batchPredictButton =
    document.getElementById("batch-predict-button");

const batchResult =
    document.getElementById("batch-result");

const downloadButton =
    document.getElementById("download-button");

const batchSummary =
    document.getElementById("batch-summary");

const batchTableBody =
    document.getElementById("batch-table-body");


// هنخزن فيه لينك الملف المؤقت
let downloadURL = null;


// =========================
// Navigation
// =========================

// Open Single Customer
singleButton.addEventListener("click", function () {

    welcomeSection.classList.add("hidden");

    singleSection.classList.remove("hidden");

});


// Back From Single Customer
backFromSingle.addEventListener("click", function () {

    singleSection.classList.add("hidden");

    welcomeSection.classList.remove("hidden");

});


// Open Whole CSV
batchButton.addEventListener("click", function () {

    welcomeSection.classList.add("hidden");

    batchSection.classList.remove("hidden");

});


// Back From Whole CSV
backFromBatch.addEventListener("click", function () {

    batchSection.classList.add("hidden");

    welcomeSection.classList.remove("hidden");

});


// =========================
// Single Customer Prediction
// =========================

predictButton.addEventListener("click", async function () {

    const tenureValue =
        document.getElementById("tenure").value;

    const monthlyChargesValue =
        document.getElementById("monthlyCharges").value;

    const totalChargesValue =
        document.getElementById("totalCharges").value;


    // Check numeric fields
    if (
        tenureValue === "" ||
        monthlyChargesValue === "" ||
        totalChargesValue === ""
    ) {

        alert("Please fill in all numeric fields.");

        return;
    }


    // Build Customer Object
    const customerData = {

        gender:
            document.getElementById("gender").value,

        SeniorCitizen:
            Number(
                document.getElementById("seniorCitizen").value
            ),

        Partner:
            document.getElementById("partner").value,

        Dependents:
            document.getElementById("dependents").value,

        tenure:
            Number(tenureValue),

        PhoneService:
            document.getElementById("phoneService").value,

        MultipleLines:
            document.getElementById("multipleLines").value,

        InternetService:
            document.getElementById("internetService").value,

        OnlineSecurity:
            document.getElementById("onlineSecurity").value,

        OnlineBackup:
            document.getElementById("onlineBackup").value,

        DeviceProtection:
            document.getElementById("deviceProtection").value,

        TechSupport:
            document.getElementById("techSupport").value,

        StreamingTV:
            document.getElementById("streamingTV").value,

        StreamingMovies:
            document.getElementById("streamingMovies").value,

        Contract:
            document.getElementById("contract").value,

        PaperlessBilling:
            document.getElementById("paperlessBilling").value,

        PaymentMethod:
            document.getElementById("paymentMethod").value,

        MonthlyCharges:
            Number(monthlyChargesValue),

        TotalCharges:
            Number(totalChargesValue)
    };


    try {

        predictButton.disabled = true;

        predictButton.textContent =
            "Predicting...";


        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(customerData)

        });


        if (!response.ok) {

            throw new Error(
                "Prediction request failed"
            );

        }


        const result =
            await response.json();


        resultPrediction.textContent =
            result.prediction;


        resultProbability.textContent =
            (result.probability * 100).toFixed(1) + "%";


        resultRisk.textContent =
            result.risk_level;


        resultCard.classList.remove("hidden");


    } catch (error) {

        console.error(error);

        alert(
            "Couldn't complete the prediction."
        );


    } finally {

        predictButton.disabled = false;

        predictButton.textContent =
            "Predict Customer";

    }

});


// =========================
// CSV File Selection
// =========================

csvFileInput.addEventListener("change", function () {

    const file =
        csvFileInput.files[0];


    if (!file) {

        return;

    }


    // Extra check that the file is CSV
    if (!file.name.toLowerCase().endsWith(".csv")) {

        alert("Please choose a CSV file.");

        csvFileInput.value = "";

        fileInfo.classList.add("hidden");

        batchPredictButton.disabled = true;

        return;
    }


    // Show selected file name
    fileName.textContent =
        file.name;


    fileInfo.classList.remove("hidden");


    // Enable prediction button
    batchPredictButton.disabled = false;


    // Hide old result if a new file was selected
    batchResult.classList.add("hidden");

});


// =========================
// Batch CSV Prediction
// =========================

batchPredictButton.addEventListener(
    "click",
    async function () {

        const file =
            csvFileInput.files[0];


        if (!file) {

            alert(
                "Please choose a CSV file."
            );

            return;
        }


        // FormData is used for sending files
        const formData =
            new FormData();


        // "file" must match FastAPI:
        // predict_batch(file: UploadFile ...)
        formData.append(
            "file",
            file
        );


        try {

            batchPredictButton.disabled = true;

            batchPredictButton.textContent =
                "Processing...";


            const response = await fetch(
                "/predict-batch",
                {

                    method: "POST",

                    body: formData

                }
            );


            if (!response.ok) {

                throw new Error(
                    "Batch prediction failed"
                );

            }


            // API returns CSV
            const csvText =
                await response.text();

            // Render Results into Table
            renderBatchResults(csvText);

            // Create blob for download button
            const csvBlob = new Blob([csvText], {
                type: "text/csv"
            });

            // Remove old temporary URL
            if (downloadURL) {
                URL.revokeObjectURL(downloadURL);
            }

            // Create temporary browser URL
            downloadURL = URL.createObjectURL(csvBlob);

            // Give the download button the file URL
            downloadButton.href = downloadURL;

            // Show result section
            batchResult.classList.remove("hidden");

        } catch (error) {

            console.error(error);

            alert(
                "Couldn't complete the batch prediction."
            );

        } finally {

            batchPredictButton.disabled = false;

            batchPredictButton.textContent =
                "Run Batch Prediction";

        }

    }
);


// =========================
// Helper: Parse CSV & Render Table
// =========================

function parseCSVLine(line) {
    const values = [];
    let cur = "";
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === "," && !inQuotes) {
            values.push(cur.trim());
            cur = "";
        } else {
            cur += char;
        }
    }
    values.push(cur.trim());
    return values;
}

function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/);
    if (lines.length < 2) {
        return { headers: [], rows: [] };
    }

    const headers = parseCSVLine(lines[0]);
    const rows = [];

    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line) {
            rows.push(parseCSVLine(line));
        }
    }

    return { headers, rows };
}

function renderBatchResults(csvText) {
    if (!batchTableBody) return;

    const { headers, rows } = parseCSV(csvText);

    if (rows.length === 0) {
        batchTableBody.innerHTML =
            '<tr><td colspan="4" style="text-align:center; padding: 20px;">No records found.</td></tr>';
        return;
    }

    const lowerHeaders = headers.map(h => h.trim().toLowerCase());

    let idCol = lowerHeaders.indexOf("customerid");
    if (idCol === -1) idCol = lowerHeaders.indexOf("id");
    if (idCol === -1) idCol = 0;

    const predCol = lowerHeaders.indexOf("prediction");
    const probCol = lowerHeaders.indexOf("probability");
    const riskCol = lowerHeaders.indexOf("risk_level");

    let churnCount = 0;
    let rowsHtml = "";

    rows.forEach((row, idx) => {
        const custId = (idCol !== -1 && row[idCol]) ? row[idCol] : `#${idx + 1}`;
        const pred = predCol !== -1 ? row[predCol] : "-";
        const probRaw = probCol !== -1 ? parseFloat(row[probCol]) : NaN;
        const probDisplay = !isNaN(probRaw) ? (probRaw * 100).toFixed(1) + "%" : (row[probCol] || "-");
        const risk = riskCol !== -1 ? row[riskCol] : "-";

        const isChurn = pred.toLowerCase() === "churn" || pred.toLowerCase() === "yes";
        if (isChurn) churnCount++;

        const predBadgeClass = isChurn ? "badge-churn" : "badge-nochurn";
        const riskLower = risk.toLowerCase();
        const riskBadgeClass =
            riskLower === "high"
                ? "badge-risk-high"
                : riskLower === "medium"
                ? "badge-risk-medium"
                : "badge-risk-low";

        rowsHtml += `
            <tr>
                <td class="cell-id"><strong>${custId}</strong></td>
                <td><span class="badge ${predBadgeClass}">${pred}</span></td>
                <td>${probDisplay}</td>
                <td><span class="badge ${riskBadgeClass}">${risk}</span></td>
            </tr>
        `;
    });

    batchTableBody.innerHTML = rowsHtml;

    if (batchSummary) {
        batchSummary.textContent =
            `Processed ${rows.length} customers (${churnCount} churn, ${rows.length - churnCount} retained). You can review all rows below or download the CSV.`;
    }
}