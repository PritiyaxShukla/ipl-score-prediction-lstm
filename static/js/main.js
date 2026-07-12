/**
 * IPL Score Predictor — Main JavaScript
 * Handles tab navigation, UI state controls, AJAX prediction requests, and SVG chart rendering.
 */

document.addEventListener("DOMContentLoaded", () => {
    // UI State variables
    let teams = [];
    let venues = [];
    let detailedOvers = []; // Array of {over_num, runs_in_over, wickets_in_over}
    let debounceTimer;

    // DOM Selectors
    const battingSelect = document.getElementById("batting-team");
    const bowlingSelect = document.getElementById("bowling-team");
    const venueSelect = document.getElementById("match-venue");
    const teamError = document.getElementById("team-validation-error");
    
    // Tab Elements
    const navLinks = document.querySelectorAll(".nav-link");
    const tabPanels = document.querySelectorAll(".tab-panel");
    
    // Mode toggles
    const radioQuick = document.getElementById("radio-quick");
    const radioDetailed = document.getElementById("radio-detailed");
    const quickInputPanel = document.getElementById("quick-input-panel");
    const detailedInputPanel = document.getElementById("detailed-input-panel");
    
    // Quick tracker inputs
    const inputOvers = document.getElementById("input-overs");
    const valDisplayOvers = document.getElementById("val-display-overs");
    const inputRuns = document.getElementById("input-runs");
    const inputWickets = document.getElementById("input-wickets");
    const quickCrr = document.getElementById("quick-crr");
    const quickWicketsRem = document.getElementById("quick-wickets-rem");
    
    // Step buttons
    const btnRunsMinus = document.getElementById("btn-runs-minus");
    const btnRunsPlus = document.getElementById("btn-runs-plus");
    const btnWicketsMinus = document.getElementById("btn-wickets-minus");
    const btnWicketsPlus = document.getElementById("btn-wickets-plus");

    // Detailed table inputs
    const oversTableBody = document.getElementById("overs-table-body");
    const tableEmptyState = document.getElementById("table-empty-state");
    const btnClearOvers = document.getElementById("btn-clear-overs");
    const btnAddOverRow = document.getElementById("btn-add-over-row");
    const btnLoadMockLog = document.getElementById("btn-load-mock-log");
    
    // Output fields
    const predictedScoreVal = document.getElementById("predicted-score-val");
    const valBarLstm = document.getElementById("val-bar-lstm");
    const fillBarLstm = document.getElementById("fill-bar-lstm");
    const valBarLinear = document.getElementById("val-bar-linear");
    const fillBarLinear = document.getElementById("fill-bar-linear");
    const currentRrVal = document.getElementById("current-rr-val");
    const biasVal = document.getElementById("bias-val");
    const apiErrorBanner = document.getElementById("api-error-banner");
    const apiErrorMsg = document.getElementById("api-error-msg");
    
    // Chart SVG Elements
    const chartGrid = document.getElementById("chart-grid");
    const chartLstmPath = document.getElementById("chart-lstm-path");
    const chartLstmPathArea = document.getElementById("chart-lstm-path-area");
    const chartLinearPath = document.getElementById("chart-linear-path");
    const chartMarkers = document.getElementById("chart-markers");
    const chartTooltip = document.getElementById("chart-tooltip");
    const svgChartContainer = document.getElementById("svg-chart-container");

    // 1. Initialization: Fetch Teams & Venues
    function init() {
        fetch("/teams-venues")
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    teams = data.teams;
                    venues = data.venues;
                    
                    // Populate select options
                    populateDropdown(battingSelect, teams);
                    populateDropdown(bowlingSelect, teams);
                    populateDropdown(venueSelect, venues);

                    // Load a default preset to make the UI immediately active and interesting
                    loadPreset("Mumbai Indians", "Chennai Super Kings", "Wankhede Stadium");
                }
            })
            .catch(err => console.error("Error loading team/venue lists:", err));
            
        setupEventListeners();
    }

    function populateDropdown(selectElement, list) {
        const placeholder = selectElement.options[0];
        selectElement.innerHTML = "";
        selectElement.appendChild(placeholder);
        
        list.forEach(item => {
            const option = document.createElement("option");
            option.value = item;
            option.textContent = item;
            selectElement.appendChild(option);
        });
    }

    // 2. Set Up Event Listeners
    function setupEventListeners() {
        // Light/Dark theme toggle (theme itself is applied pre-paint by the inline
        // script in index.html; this just handles the click + persistence)
        const themeToggleBtn = document.getElementById("theme-toggle-btn");
        if (themeToggleBtn) {
            themeToggleBtn.addEventListener("click", () => {
                const root = document.documentElement;
                const next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
                root.setAttribute("data-theme", next);
                localStorage.setItem("ipl-theme", next);
            });
        }

        // Tab Switcher logic
        navLinks.forEach(link => {
            link.addEventListener("click", (e) => {
                const targetTab = e.currentTarget.dataset.tab;
                
                // Toggle active state on links
                navLinks.forEach(l => l.classList.remove("active"));
                e.currentTarget.classList.add("active");
                
                // Toggle active state on panels
                tabPanels.forEach(panel => {
                    if (panel.id === `tab-${targetTab}`) {
                        panel.classList.remove("hidden");
                    } else {
                        panel.classList.add("hidden");
                    }
                });

                // Redraw chart if returning to Simulator tab
                if (targetTab === "simulator" && window.lastPredictionData) {
                    // Let the DOM update first, then draw
                    setTimeout(() => {
                        renderChart(window.lastPredictionData.full_sequence);
                    }, 50);
                }
            });
        });

        // Dropdown changes trigger verification and predictions
        battingSelect.addEventListener("change", handleMatchConfigChange);
        bowlingSelect.addEventListener("change", handleMatchConfigChange);
        venueSelect.addEventListener("change", () => triggerPredictionDebounced());
        
        // Listen to prediction type radios (Projected vs Raw)
        document.querySelectorAll('input[name="prediction_type"]').forEach(radio => {
            radio.addEventListener("change", () => triggerPredictionDebounced());
        });

        // Preset button clicks
        document.querySelectorAll(".btn-preset").forEach(btn => {
            btn.addEventListener("click", (e) => {
                document.querySelectorAll(".btn-preset").forEach(b => b.classList.remove("active"));
                
                const presetBtn = e.currentTarget;
                presetBtn.classList.add("active");
                
                const batting = presetBtn.dataset.batting;
                const bowling = presetBtn.dataset.bowling;
                const venue = presetBtn.dataset.venue;
                loadPreset(batting, bowling, venue);
            });
        });

        // Mode pills toggle
        radioQuick.addEventListener("change", handleModeChange);
        radioDetailed.addEventListener("change", handleModeChange);

        // Quick inputs listeners
        inputOvers.addEventListener("input", (e) => {
            valDisplayOvers.textContent = parseFloat(e.target.value).toFixed(1);
            updateQuickCRR();
            triggerPredictionDebounced();
        });
        
        inputRuns.addEventListener("input", (e) => {
            let val = parseInt(e.target.value);
            if (!isNaN(val)) {
                if (val > 350) e.target.value = 350;
                if (val < 0) e.target.value = 0;
            }
            updateQuickCRR();
            triggerPredictionDebounced();
        });
        
        inputWickets.addEventListener("input", (e) => {
            let val = parseInt(e.target.value);
            if (!isNaN(val)) {
                if (val > 10) e.target.value = 10;
                if (val < 0) e.target.value = 0;
            }
            updateQuickWicketsDisplay();
            triggerPredictionDebounced();
        });

        // Step buttons for runs and wickets
        btnRunsMinus.addEventListener("click", () => adjustNumericInput(inputRuns, -5));
        btnRunsPlus.addEventListener("click", () => adjustNumericInput(inputRuns, 5));
        btnWicketsMinus.addEventListener("click", () => adjustNumericInput(inputWickets, -1));
        btnWicketsPlus.addEventListener("click", () => adjustNumericInput(inputWickets, 1));

        // Detailed mode actions
        btnAddOverRow.addEventListener("click", () => addOverRow());
        btnClearOvers.addEventListener("click", resetDetailedLog);
        btnLoadMockLog.addEventListener("click", loadMockLogData);
        
        // Handle window resizing to redraw chart
        window.addEventListener("resize", () => {
            if (window.lastPredictionData) {
                renderChart(window.lastPredictionData.full_sequence);
            }
        });
    }

    // 3. Match Config / Preset Handlers
    function loadPreset(batting, bowling, venue) {
        battingSelect.value = batting;
        bowlingSelect.value = bowling;
        venueSelect.value = venue;
        
        verifyTeamsDifferent();
        triggerPredictionDebounced();
    }

    function handleMatchConfigChange() {
        if (verifyTeamsDifferent()) {
            document.querySelectorAll(".btn-preset").forEach(b => b.classList.remove("active"));
            triggerPredictionDebounced();
        }
    }

    function verifyTeamsDifferent() {
        const batting = battingSelect.value;
        const bowling = bowlingSelect.value;
        
        if (batting && bowling && batting === bowling) {
            teamError.classList.add("show");
            predictedScoreVal.textContent = "—";
            return false;
        } else {
            teamError.classList.remove("show");
            return true;
        }
    }

    // 4. Input Mode Switcher (Quick vs Detailed)
    function handleModeChange() {
        if (radioQuick.checked) {
            quickInputPanel.classList.remove("hidden");
            detailedInputPanel.classList.add("hidden");
        } else {
            quickInputPanel.classList.add("hidden");
            detailedInputPanel.classList.remove("hidden");
            if (detailedOvers.length === 0) {
                loadMockLogData();
                return;
            }
        }
        triggerPredictionDebounced();
    }

    // 5. Quick Mode Helpers
    function adjustNumericInput(inputElement, step) {
        const currentVal = parseInt(inputElement.value) || 0;
        const minVal = parseInt(inputElement.min) || 0;
        const maxVal = parseInt(inputElement.max) || 300;
        
        let newVal = currentVal + step;
        if (newVal < minVal) newVal = minVal;
        if (newVal > maxVal) newVal = maxVal;
        
        inputElement.value = newVal;
        
        if (inputElement === inputRuns) {
            updateQuickCRR();
        } else if (inputElement === inputWickets) {
            updateQuickWicketsDisplay();
        }
        
        triggerPredictionDebounced();
    }

    function updateQuickCRR() {
        const runs = parseFloat(inputRuns.value) || 0;
        const overs = parseFloat(inputOvers.value) || 1;
        const crr = runs / overs;
        quickCrr.textContent = crr.toFixed(2);
    }

    function updateQuickWicketsDisplay() {
        const wickets = parseInt(inputWickets.value) || 0;
        if (wickets >= 10) {
            quickWicketsRem.textContent = "All Out";
        } else {
            quickWicketsRem.textContent = 10 - wickets;
        }
    }

    // 6. Detailed Mode Log Manager
    function renderDetailedTable() {
        oversTableBody.innerHTML = "";
        
        if (detailedOvers.length === 0) {
            tableEmptyState.classList.remove("hidden");
            return;
        }
        
        tableEmptyState.classList.add("hidden");
        
        let cumRuns = 0;
        let cumWickets = 0;
        
        detailedOvers.forEach((over, index) => {
            over.over_num = index + 1;
            cumRuns += over.runs_in_over;
            cumWickets += over.wickets_in_over;
            
            const overRr = cumRuns / over.over_num;
            
            const row = document.createElement("tr");
            row.innerHTML = `
                <td><strong>Over ${over.over_num}</strong></td>
                <td class="table-input-cell">
                    <input type="number" class="table-input cell-runs-input" min="0" max="36" value="${over.runs_in_over}" data-index="${index}">
                </td>
                <td><span>${cumRuns}</span></td>
                <td class="table-input-cell">
                    <input type="number" class="table-input cell-wickets-input" min="0" max="6" value="${over.wickets_in_over}" data-index="${index}">
                </td>
                <td><span>${cumWickets}</span></td>
                <td><strong class="text-secondary">${overRr.toFixed(2)}</strong></td>
                <td class="actions-col">
                    <button type="button" class="btn-delete" data-index="${index}" aria-label="Delete over">×</button>
                </td>
            `;
            
            row.querySelector(".cell-runs-input").addEventListener("input", handleCellChange);
            row.querySelector(".cell-wickets-input").addEventListener("input", handleCellChange);
            row.querySelector(".btn-delete").addEventListener("click", handleDeleteOver);
            
            oversTableBody.appendChild(row);
        });
    }

    function addOverRow() {
        if (detailedOvers.length >= 19) {
            alert("Maximum 19 overs can be logged. Predictor runs on standard 20-over T20 matches.");
            return;
        }
        
        detailedOvers.push({
            runs_in_over: 6,
            wickets_in_over: 0
        });
        
        renderDetailedTable();
        triggerPredictionDebounced();
    }

    function handleCellChange(e) {
        const index = parseInt(e.target.dataset.index);
        const value = parseInt(e.target.value) || 0;
        
        if (e.target.classList.contains("cell-runs-input")) {
            detailedOvers[index].runs_in_over = Math.max(0, Math.min(36, value));
        } else {
            detailedOvers[index].wickets_in_over = Math.max(0, Math.min(6, value));
        }
        
        let totalWickets = 0;
        for (let i = 0; i < detailedOvers.length; i++) {
            totalWickets += detailedOvers[i].wickets_in_over;
            if (totalWickets > 10) {
                detailedOvers[i].wickets_in_over -= (totalWickets - 10);
                if (detailedOvers[i].wickets_in_over < 0) detailedOvers[i].wickets_in_over = 0;
                totalWickets = 10;
            }
        }
        
        renderDetailedTable();
        triggerPredictionDebounced();
    }

    function handleDeleteOver(e) {
        const index = parseInt(e.currentTarget.dataset.index);
        detailedOvers.splice(index, 1);
        renderDetailedTable();
        triggerPredictionDebounced();
    }

    function resetDetailedLog() {
        detailedOvers = [];
        renderDetailedTable();
        triggerPredictionDebounced();
    }

    function loadMockLogData() {
        detailedOvers = [
            { runs_in_over: 8, wickets_in_over: 0 },
            { runs_in_over: 7, wickets_in_over: 1 },
            { runs_in_over: 9, wickets_in_over: 0 },
            { runs_in_over: 12, wickets_in_over: 0 },
            { runs_in_over: 6, wickets_in_over: 0 },
            { runs_in_over: 14, wickets_in_over: 0 },
            { runs_in_over: 5, wickets_in_over: 1 },
            { runs_in_over: 8, wickets_in_over: 0 },
            { runs_in_over: 11, wickets_in_over: 0 },
            { runs_in_over: 6, wickets_in_over: 1 }
        ];
        
        renderDetailedTable();
        triggerPredictionDebounced();
    }

    // 7. Prediction Caller
    function triggerPredictionDebounced() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(callPredictionAPI, 250);
    }

    function callPredictionAPI() {
        const batting = battingSelect.value;
        const bowling = bowlingSelect.value;
        const venue = venueSelect.value;
        const predictionType = document.querySelector('input[name="prediction_type"]:checked').value;
        
        if (!batting || !bowling || !venue || batting === bowling) {
            return;
        }

        const isQuickMode = radioQuick.checked;
        let payload = {
            batting_team: batting,
            bowling_team: bowling,
            venue: venue,
            prediction_type: predictionType,
            mode: isQuickMode ? "quick" : "detailed"
        };

        if (isQuickMode) {
            payload.current_over = parseFloat(inputOvers.value);
            payload.current_runs = parseFloat(inputRuns.value);
            payload.current_wickets = parseFloat(inputWickets.value);
        } else {
            let cumRuns = 0;
            let cumWickets = 0;
            payload.overs_so_far = detailedOvers.map((over, index) => {
                cumRuns += over.runs_in_over;
                cumWickets += over.wickets_in_over;
                return {
                    current_over: index + 1,
                    cum_runs: cumRuns,
                    cum_wickets: cumWickets,
                    current_run_rate: cumRuns / (index + 1)
                };
            });
        }

        predictedScoreVal.textContent = "...";
        apiErrorBanner.classList.add("hidden");
        
        fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                window.lastPredictionData = data;
                apiErrorBanner.classList.add("hidden");
                
                // Only update the active UI elements if we are on the Simulator panel
                const activeTab = document.querySelector(".nav-link.active").dataset.tab;
                if (activeTab === "simulator") {
                    updatePredictionUI(data);
                }
            } else {
                predictedScoreVal.textContent = "—";
                fillBarLstm.style.width = "0%";
                fillBarLinear.style.width = "0%";
                apiErrorMsg.textContent = data.error;
                apiErrorBanner.classList.remove("hidden");
                console.error("Prediction failed:", data.error);
            }
        })
        .catch(err => {
            predictedScoreVal.textContent = "—";
            fillBarLstm.style.width = "0%";
            fillBarLinear.style.width = "0%";
            apiErrorMsg.textContent = "Failed to connect to the backend server.";
            apiErrorBanner.classList.remove("hidden");
            console.error("API request failed:", err);
        });
    }

    // 8. Update UI Elements
    function updatePredictionUI(data) {
        const predictedScore = data.predicted_score;
        const linearProject = data.linear_projection;
        
        predictedScoreVal.textContent = predictedScore.toFixed(1);
        currentRrVal.textContent = data.current_run_rate.toFixed(2);
        
        const difference = predictedScore - linearProject;
        let biasText = "Neutral";
        if (difference > 8) {
            biasText = "Batting Advantage";
        } else if (difference > 2) {
            biasText = "Slight Batting Bias";
        } else if (difference < -8) {
            biasText = "Bowling Advantage";
        } else if (difference < -2) {
            biasText = "Slight Bowling Bias";
        }
        biasVal.textContent = biasText;
        
        const scaleMax = 250;
        
        valBarLstm.textContent = `${predictedScore.toFixed(1)} Runs`;
        const lstmPercent = Math.min(100, Math.max(0, (predictedScore / scaleMax) * 100));
        fillBarLstm.style.width = `${lstmPercent}%`;
        
        valBarLinear.textContent = `${linearProject.toFixed(1)} Runs`;
        const linearPercent = Math.min(100, Math.max(0, (linearProject / scaleMax) * 100));
        fillBarLinear.style.width = `${linearPercent}%`;

        renderChart(data.full_sequence);
    }

    // 9. Interactive SVG Chart Drawer (Indigo Palette)
    function renderChart(sequence) {
        if (!sequence || sequence.length === 0) return;
        
        // Reset container dimensions
        const width = 500;
        const height = 220;
        const paddingLeft = 30;
        const paddingRight = 15;
        const paddingTop = 15;
        const paddingBottom = 20;
        
        const chartWidth = width - paddingLeft - paddingRight;
        const chartHeight = height - paddingTop - paddingBottom;
        
        chartGrid.innerHTML = "";
        chartMarkers.innerHTML = "";
        
        let maxRuns = 180;
        sequence.forEach(over => {
            if (over.cum_runs > maxRuns) {
                maxRuns = Math.ceil(over.cum_runs / 10) * 10;
            }
        });
        
        const oversCount = 20;
        
        // X Axis grid & labels
        for (let i = 0; i <= oversCount; i += 5) {
            const xVal = paddingLeft + (i / oversCount) * chartWidth;
            
            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line.setAttribute("x1", xVal);
            line.setAttribute("y1", paddingTop);
            line.setAttribute("x2", xVal);
            line.setAttribute("y2", paddingTop + chartHeight);
            line.setAttribute("class", "chart-grid-line");
            chartGrid.appendChild(line);
            
            const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
            text.setAttribute("x", xVal);
            text.setAttribute("y", paddingTop + chartHeight + 15);
            text.setAttribute("text-anchor", "middle");
            text.setAttribute("class", "chart-label");
            text.textContent = i === 0 ? "Start" : `${i} Ov`;
            chartGrid.appendChild(text);
        }
        
        // Y Axis grid & labels
        for (let r = 0; r <= maxRuns; r += 50) {
            const yVal = paddingTop + chartHeight - (r / maxRuns) * chartHeight;
            
            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line.setAttribute("x1", paddingLeft);
            line.setAttribute("y1", yVal);
            line.setAttribute("x2", paddingLeft + chartWidth);
            line.setAttribute("y2", yVal);
            line.setAttribute("class", "chart-grid-line");
            chartGrid.appendChild(line);
            
            const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
            text.setAttribute("x", paddingLeft - 8);
            text.setAttribute("y", yVal + 3);
            text.setAttribute("text-anchor", "end");
            text.setAttribute("class", "chart-label");
            text.textContent = r;
            chartGrid.appendChild(text);
        }

        // Draw Axis Bounds
        const baselineX = document.createElementNS("http://www.w3.org/2000/svg", "line");
        baselineX.setAttribute("x1", paddingLeft);
        baselineX.setAttribute("y1", paddingTop + chartHeight);
        baselineX.setAttribute("x2", paddingLeft + chartWidth);
        baselineX.setAttribute("y2", paddingTop + chartHeight);
        baselineX.setAttribute("class", "chart-axis-line");
        chartGrid.appendChild(baselineX);
        
        const baselineY = document.createElementNS("http://www.w3.org/2000/svg", "line");
        baselineY.setAttribute("x1", paddingLeft);
        baselineY.setAttribute("y1", paddingTop);
        baselineY.setAttribute("x2", paddingLeft);
        baselineY.setAttribute("y2", paddingTop + chartHeight);
        baselineY.setAttribute("class", "chart-axis-line");
        chartGrid.appendChild(baselineY);

        // Build Paths
        let lstmPoints = [];
        let linearPoints = [];
        
        lstmPoints.push({x: paddingLeft, y: paddingTop + chartHeight});
        linearPoints.push({x: paddingLeft, y: paddingTop + chartHeight});

        sequence.forEach((over, index) => {
            const x = paddingLeft + (over.current_over / oversCount) * chartWidth;
            const yLstm = paddingTop + chartHeight - (over.cum_runs / maxRuns) * chartHeight;
            lstmPoints.push({x: x, y: yLstm, over: over.current_over, runs: over.cum_runs, wickets: over.cum_wickets});
            
            const finalLinearScore = window.lastPredictionData ? window.lastPredictionData.linear_projection : 0;
            const yLinearVal = over.current_over * (finalLinearScore / 20.0);
            const yLinear = paddingTop + chartHeight - (yLinearVal / maxRuns) * chartHeight;
            linearPoints.push({x: x, y: yLinear});
        });

        // Draw linear path
        let linearD = `M ${linearPoints[0].x} ${linearPoints[0].y} `;
        for (let i = 1; i < linearPoints.length; i++) {
            linearD += `L ${linearPoints[i].x} ${linearPoints[i].y} `;
        }
        chartLinearPath.setAttribute("d", linearD);

        // Draw LSTM solid line & fill area
        let lstmD = `M ${lstmPoints[0].x} ${lstmPoints[0].y} `;
        let areaD = `M ${lstmPoints[0].x} ${lstmPoints[0].y} `;
        
        for (let i = 1; i < lstmPoints.length; i++) {
            lstmD += `L ${lstmPoints[i].x} ${lstmPoints[i].y} `;
            areaD += `L ${lstmPoints[i].x} ${lstmPoints[i].y} `;
        }
        
        areaD += `L ${lstmPoints[lstmPoints.length - 1].x} ${paddingTop + chartHeight} L ${paddingLeft} ${paddingTop + chartHeight} Z`;
        
        chartLstmPath.setAttribute("d", lstmD);
        chartLstmPathArea.setAttribute("d", areaD);

        // Add Markers & Hover states
        let prevWickets = 0;
        
        lstmPoints.forEach((pt, index) => {
            if (index === 0) return;
            
            const currentWickets = pt.wickets;
            const wicketFell = currentWickets > prevWickets;
            
            const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            dot.setAttribute("cx", pt.x);
            dot.setAttribute("cy", pt.y);
            
            if (wicketFell) {
                dot.setAttribute("r", 4.5);
                dot.setAttribute("fill", "#f43f5e"); // Rose 500
                dot.setAttribute("stroke", "#ffffff");
                dot.setAttribute("stroke-width", "1.25");
                dot.setAttribute("class", "wicket-marker");
            } else {
                dot.setAttribute("r", 2.5);
                dot.setAttribute("fill", "#6366f1"); // Indigo 500
                dot.setAttribute("opacity", "0");
                dot.setAttribute("class", "hover-marker");
            }
            
            dot.addEventListener("mouseenter", (e) => {
                dot.setAttribute("r", wicketFell ? 6.5 : 4.5);
                if (!wicketFell) dot.setAttribute("opacity", "1");
                showTooltip(e, pt.over, pt.runs, pt.wickets, wicketFell);
            });
            
            dot.addEventListener("mouseleave", () => {
                dot.setAttribute("r", wicketFell ? 4.5 : 2.5);
                if (!wicketFell) dot.setAttribute("opacity", "0");
                hideTooltip();
            });

            chartMarkers.appendChild(dot);
            prevWickets = currentWickets;
        });
    }

    function showTooltip(event, over, runs, wickets, wicketFell) {
        const containerRect = svgChartContainer.getBoundingClientRect();
        const mouseX = event.clientX - containerRect.left;
        const mouseY = event.clientY - containerRect.top;
        
        let content = `
            <strong>Over ${over}</strong><br/>
            Score: ${runs.toFixed(1)} / ${wickets}
        `;
        if (wicketFell) {
            content += `<br/><span style="color:#f43f5e; font-weight:bold;">Wicket Lost</span>`;
        }
        
        chartTooltip.innerHTML = content;
        chartTooltip.style.display = "block";
        chartTooltip.style.left = `${mouseX + 10}px`;
        chartTooltip.style.top = `${mouseY - 20}px`;
    }

    function hideTooltip() {
        chartTooltip.style.display = "none";
    }

    init();
});
