// const items = document.querySelectorAll('.carousel-item');
// const nextBtn = document.getElementById('nextBtn');
// console.log(items.length);

// let currentIndex = 0;

// // Function to show a specific slide
// function showSlide(index) {
//   // 1. Add 'hidden' class from the current item
//   items[currentIndex].classList.add('hidden');
  
//   // 2. Update index (wrap around if at the end)
//   currentIndex = (index + items.length) % items.length;
  
//   // 3. Remove 'hidden' class to the new item
//   items[currentIndex].classList.remove('hidden');
// }

// // Auto-rotate every 5 seconds (Optional)
// setInterval(() => {
//   showSlide(currentIndex + 1);
// }, 5000);

// // Button Click Event
// nextBtn.addEventListener('click', () => {
//   showSlide(currentIndex + 1);
// });

const items = document.querySelectorAll('.carousel-item');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const counter = document.querySelector('.slide-counter');
const pplEl = document.getElementById('fmlm-ppl');
const entropyEl = document.getElementById('fmlm-entropy');

let currentIndex = 0;

function updateCarousel(newIndex) {
  // 1. Hide current item
  items[currentIndex].classList.add('hidden');
  
  // 2. Update Index (Handle wrapping for both directions)
  currentIndex = (newIndex + items.length) % items.length;
  
  // 3. Show new item
  items[currentIndex].classList.remove('hidden');

  // 4. Update the counter text
  if(counter) {
      counter.textContent = `Sample ${currentIndex + 1} / ${items.length}`;
  }

  // 5. Update per-sample metrics (Gen. PPL & Entropy)
  const currentItem = items[currentIndex];
  if (pplEl && currentItem.dataset.ppl) {
      pplEl.textContent = currentItem.dataset.ppl;
  }
  if (entropyEl && currentItem.dataset.entropy) {
      entropyEl.textContent = currentItem.dataset.entropy;
  }
}

// Initialize counter on page load
if (counter) {
  counter.textContent = `Sample ${currentIndex + 1} / ${items.length}`;
}

// Event Listeners
nextBtn.addEventListener('click', () => {
  updateCarousel(currentIndex + 1);
});

prevBtn.addEventListener('click', () => {
  updateCarousel(currentIndex - 1);
});