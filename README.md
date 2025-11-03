# The GAIEA (Global AI Efficiency Agency) Policy Simulator

AI is using a ton of energy and water. How do we fix it without just banning things? This project is a Python-based simulator built to test one possible solution: **The GAIEA**.

This model moves beyond a simple "carbon tax" and simulates a **transparency-based policy**. The core idea is: what if an independent agency audited AI models and gave them a public "grade" based on their environmental impact?

## 💡 The Policy Idea: GAIEA & The EER Rating

This simulator is built on a hypothetical policy idea I developed called the **Global AI Efficiency Agency (GAIEA)**. Its job is simple:

1.  **Audit:** The agency independently audits frontier AI models for their true environmental cost (energy use, water use, % renewables).
2.  **Rate:** Based on this audit, it gives the model a public **"EER" (Environmental Efficiency Rating)** from A++ (most efficient) to F (most wasteful).
3.  **Drive Market Change:** The simulation assumes that companies would compete for a better EER grade. An "A" rating becomes a marketing advantage, forcing AI labs to invest in efficiency and renewable energy.

## 🚀 How to Use the Simulator

This is a simple command-line tool. You don't need to edit the code to use it.


Make sure you have the required Python libraries.
```bash
pip install matplotlib numpy

python gaiea_simulator.py

Note You might need to replace gaiea_simulator.py with whatever you named your file.)

3. Answer the Prompts: The script will ask you for two types of inputs:

Initial Conditions (2025): What's the starting point? (e.g., initial energy use, water use, renewable %).

Policy Levers:

Activate GAIEA Policy? (y/n): The main on/off switch.

Market Pressure Factor (1-5): The most important input! This is your guess for how much the market cares about the EER grade. A '5' means high competition.

📊 What You Get
After you answer the prompts, the simulator will give you two outputs:

A "Report Card" in your Terminal: It prints a full analysis, including your scenario's final Projected EER Rating (A++ to F), required policy actions, and smart "Rerun Suggestions" to help you find a path to a better grade.

A 4-Panel Graph: A pop-up window (using Matplotlib) shows the full forecast for:

Energy Consumption

Water Consumption

Carbon Emissions (compared to "Business As Usual")

Renewable & Efficiency Gains

🔬 Test It Yourself
A great way to see the model in action is to run two different scenarios:

Scenario 1: No Policy

Activate GAIEA Policy?: n

Observe the high growth in emissions and the low EER grade.

Scenario 2: Strong Policy

Activate GAIEA Policy?: y

Market Pressure Factor: 5

Observe how the efficiency and renewable lines curve upwards, and the emissions line flattens out.
