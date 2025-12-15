# Water Tank Problem — Frontend Visualizer

This is a simple frontend project (Vanilla JavaScript + HTML/CSS + SVG) that visualizes the classic **Trapping Rain Water** problem.

## Features
- Parse input array of heights (comma-separated)
- Compute trapped water using an efficient two-pointer algorithm
- Show total trapped water (units)
- Draw an SVG visualization showing blocks and water
- Random example generator and reset

## How to run
1. Clone the repository or download the files.
2. Open `index.html` in your browser (no server required).

## Example
Input: `0,4,0,0,0,6,0,6,4,0`  
Output: Total trapped water units — displayed on UI and visualized in blue.

## Files
- `index.html` — main page
- `style.css` — styling
- `script.js` — logic & SVG renderer
- `README.md` — this file

## Notes
- Visualization scales by height. For very large heights, reduce values or adjust `unit` in `script.js`.
- This is a frontend-only demo suitable for interviews or quick prototyping.
