// script.js — Water Tank Problem visualiser (vanilla JS + SVG)
// Algorithm: compute trapped water using two-pointer approach and per-index water amounts

function parseInput(str){
  return str.split(',').map(s => parseInt(s.trim())).filter(x => !Number.isNaN(x) && x >= 0);
}

// Two-pointer method to get total trapped water and per-index trapped water
function computeTrapped(height){
  const n = height.length;
  let left = 0, right = n - 1;
  let leftMax = 0, rightMax = 0;
  let total = 0;
  const water = new Array(n).fill(0);

  while(left <= right){
    if(height[left] <= height[right]){
      if(height[left] >= leftMax){
        leftMax = height[left];
      } else {
        const w = leftMax - height[left];
        water[left] = w;
        total += w;
      }
      left++;
    } else {
      if(height[right] >= rightMax){
        rightMax = height[right];
      } else {
        const w = rightMax - height[right];
        water[right] = w;
        total += w;
      }
      right--;
    }
  }
  return { total, water };
}

// SVG rendering
function renderSVG(height, water){
  const padding = 12;
  const n = height.length;
  if(n === 0) return '<div style="color:#f88">No valid heights provided</div>';
  const maxH = Math.max(...height, 1);
  const unit = 24; // pixels per height unit for visualization (scaled)
  const barWidth = Math.max(24, Math.floor(800 / n));
  const svgWidth = Math.max(600, n * (barWidth + 6) + padding * 2);
  const svgHeight = maxH * unit + 160;

  let svg = '';
  svg += `<svg width="${svgWidth}" height="${svgHeight}" viewBox="0 0 ${svgWidth} ${svgHeight}" xmlns="http://www.w3.org/2000/svg">`;
  // background
  svg += `<rect x="0" y="0" width="${svgWidth}" height="${svgHeight}" fill="none" />`;

  const baseline = svgHeight - 80;
  // Draw grid / axis labels (optional)
  for(let i = 0; i <= maxH; i++){
    const y = baseline - i * unit;
    svg += `<line x1="0" x2="${svgWidth}" y1="${y}" y2="${y}" stroke="#ffffff09" />`;
    if(i % 2 === 0){
      svg += `<text x="6" y="${y - 6}" fill="#9aa7b2" font-size="11">${i}</text>`;
    }
  }

  // Draw bars and water
  const gap = 6;
  const startX = padding + 40;
  for(let i = 0; i < n; i++){
    const h = height[i];
    const w = water[i] || 0;
    const x = startX + i * (barWidth + gap);
    const barH = h * unit;
    const barY = baseline - barH;
    const waterH = w * unit;
    const waterY = baseline - (barH + waterH);

    // Bar (block)
    svg += `<rect x="${x}" y="${barY}" width="${barWidth}" height="${barH}" rx="6" fill="#222" stroke="#ffffff14" />`;
    // Bar top highlight
    svg += `<rect x="${x}" y="${barY}" width="${barWidth}" height="${Math.min(6, barH)}" fill="#ffffff08" />`;

    // Water (if any)
    if(w > 0){
      svg += `<rect x="${x}" y="${waterY}" width="${barWidth}" height="${waterH}" rx="4" fill="#3aa6ff66" />`;
      // water surface line
      svg += `<line x1="${x}" x2="${x + barWidth}" y1="${waterY}" y2="${waterY}" stroke="#3aa6ff" stroke-width="2" />`;
    }

    // index label
    svg += `<text x="${x + barWidth/2}" y="${baseline + 18}" fill="#9aa7b2" font-size="12" text-anchor="middle">${i}</text>`;
    // height label
    svg += `<text x="${x + barWidth/2}" y="${barY - 6}" fill="#dbeafe" font-size="12" text-anchor="middle">${h}</text>`;
    // water label
    if(w > 0){
      svg += `<text x="${x + barWidth/2}" y="${waterY + 16}" fill="#042026" font-size="12" font-weight="700" text-anchor="middle">${w}</text>`;
    }
  }

  svg += `</svg>`;

  // Legend
  svg += `<div class="legend"><div><span style="background:#222"></span> Block</div><div><span style="background:rgba(58,166,255,0.4)"></span> Water</div></div>`;
  return svg;
}

document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('inputArray');
  const computeBtn = document.getElementById('computeBtn');
  const randomBtn = document.getElementById('randomBtn');
  const resetBtn = document.getElementById('resetBtn');
  const viz = document.getElementById('vizContainer');
  const waterUnits = document.getElementById('waterUnits');

  function run(){
    const arr = parseInput(input.value);
    if(arr.length === 0){
      viz.innerHTML = '<div style="color:#f88">Please enter a valid comma-separated list of non-negative integers.</div>';
      waterUnits.textContent = '-';
      return;
    }
    const { total, water } = computeTrapped(arr);
    waterUnits.textContent = total;
    viz.innerHTML = renderSVG(arr, water);
  }

  computeBtn.addEventListener('click', run);

  randomBtn.addEventListener('click', () => {
    // create random example
    const n = Math.floor(Math.random()*8) + 6;
    const arr = Array.from({length:n}, () => Math.floor(Math.random()*7));
    input.value = arr.join(',');
    run();
  });

  resetBtn.addEventListener('click', () => {
    input.value = '';
    viz.innerHTML = '';
    waterUnits.textContent = '-';
  });

  // initial render
  run();
});
