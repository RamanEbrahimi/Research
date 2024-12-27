const startingFeatures = [60, 40, 60, 65];
const weights = [0.5, 0.3, 0.2, 0.4]; // Example weights
const budget = 10;
const threshold = 50;
const interactions = []; // To store interaction data

// Render the Feature Weights Chart
function renderChart() {
    const ctx = document.getElementById("weightsChart").getContext("2d");
    new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Feature 1", "Feature 2", "Feature 3", "Feature 4"],
            datasets: [
                {
                    label: "Weights",
                    data: weights,
                    backgroundColor: "rgba(75, 192, 192, 0.6)",
                    borderColor: "rgba(75, 192, 192, 1)",
                    borderWidth: 1,
                },
            ],
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                },
            },
        },
    });
}

// Calculate Prediction and Store Interaction
function calculatePrediction() {
    let allocatedHours = [
        parseInt(document.getElementById("feature1").value || 0),
        parseInt(document.getElementById("feature2").value || 0),
        parseInt(document.getElementById("feature3").value || 0),
        parseInt(document.getElementById("feature4").value || 0),
    ];

    let totalHours = allocatedHours.reduce((a, b) => a + b, 0);

    if (totalHours > budget) {
        document.getElementById("result").innerText = "Error: Budget exceeded!";
        return;
    }

    // Calculate initial score
    let initialScore = startingFeatures.reduce(
        (sum, feature, index) => sum + feature * weights[index],
        0
    );

    let updatedFeatures = startingFeatures.map((feature, index) => {
        let hours = allocatedHours[index];
        let additionalPoints = 0;

        if (hours > 4) {
            additionalPoints += 4 * 5; // First 4 hours
            hours -= 4;

            if (hours > 4) {
                additionalPoints += 4 * 2.5; // Next 4 hours
                hours -= 4;

                additionalPoints += hours * 1; // Last 2 hours
            } else {
                additionalPoints += hours * 2.5; // Remaining hours in second tier
            }
        } else {
            additionalPoints += hours * 5; // All hours in first tier
        }

        return feature + additionalPoints;
    });

    // Calculate final score
    let finalScore = updatedFeatures.reduce(
        (sum, feature, index) => sum + feature * weights[index],
        0
    );

    // Determine classification
    let classification = finalScore >= threshold ? "Approved" : "Rejected";

    // Store Interaction Data
    interactions.push({
        allocatedHours,
        initialScore: initialScore.toFixed(2),
        finalScore: finalScore.toFixed(2),
        classification,
    });

    console.log("Interactions:", interactions); // View stored interactions in the console

    // Display results
    document.getElementById("result").innerText = `
        Initial Score: ${initialScore.toFixed(2)}
        Final Score: ${finalScore.toFixed(2)}
        Threshold: ${threshold}
        Prediction: ${classification}
    `;
}

// Initialize the page
document.addEventListener("DOMContentLoaded", renderChart);
