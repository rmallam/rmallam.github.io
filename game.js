// ================================================
// CI/CD PIPELINE BUILDER GAME
// ================================================

// Game State
const gameState = {
    currentLevel: 1,
    score: 0,
    startTime: null,
    timer: null,
    pipelineStages: [],
    maxLevel: 5
};

// Level Definitions with Increasing Difficulty
const levels = [
    {
        level: 1,
        title: "Level 1: Basic Pipeline",
        description: "Build a simple CI/CD pipeline with the essential stages",
        requiredStages: ['Code', 'Build', 'Test', 'Deploy'],
        availableStages: ['Code', 'Build', 'Test', 'Deploy', 'Monitor', 'Security Scan'],
        hint: "A basic pipeline follows: Code → Build → Test → Deploy"
    },
    {
        level: 2,
        title: "Level 2: Add Security",
        description: "Enhance your pipeline with security scanning before deployment",
        requiredStages: ['Code', 'Build', 'Test', 'Security Scan', 'Deploy'],
        availableStages: ['Code', 'Build', 'Test', 'Security Scan', 'Deploy', 'Monitor', 'Package'],
        hint: "Security scans should run after tests but before deployment"
    },
    {
        level: 3,
        title: "Level 3: Complete CI/CD",
        description: "Add packaging and monitoring for a production-ready pipeline",
        requiredStages: ['Code', 'Build', 'Test', 'Package', 'Security Scan', 'Deploy', 'Monitor'],
        availableStages: ['Code', 'Build', 'Test', 'Package', 'Security Scan', 'Deploy', 'Monitor', 'Rollback', 'Approve'],
        hint: "Package your application after testing, then scan, deploy, and monitor"
    },
    {
        level: 4,
        title: "Level 4: Add Approval Gate",
        description: "Include manual approval before production deployment",
        requiredStages: ['Code', 'Build', 'Test', 'Package', 'Security Scan', 'Approve', 'Deploy', 'Monitor'],
        availableStages: ['Code', 'Build', 'Test', 'Package', 'Security Scan', 'Approve', 'Deploy', 'Monitor', 'Rollback'],
        hint: "Manual approval should come after security scan but before deploy"
    },
    {
        level: 5,
        title: "Level 5: Enterprise Pipeline",
        description: "Build a complete enterprise-grade CI/CD pipeline with rollback capability",
        requiredStages: ['Code', 'Build', 'Test', 'Package', 'Security Scan', 'Approve', 'Deploy', 'Monitor', 'Rollback'],
        availableStages: ['Code', 'Build', 'Test', 'Package', 'Security Scan', 'Approve', 'Deploy', 'Monitor', 'Rollback'],
        hint: "Complete pipeline: Code → Build → Test → Package → Security Scan → Approve → Deploy → Monitor → Rollback"
    }
];

// Stage Definitions with Icons and Descriptions
const stageDefinitions = {
    'Code': {
        icon: 'fa-code',
        color: 'code',
        description: 'Source code checkout'
    },
    'Build': {
        icon: 'fa-hammer',
        color: 'build',
        description: 'Compile and build'
    },
    'Test': {
        icon: 'fa-flask',
        color: 'test',
        description: 'Run automated tests'
    },
    'Package': {
        icon: 'fa-box',
        color: 'package',
        description: 'Package application'
    },
    'Security Scan': {
        icon: 'fa-shield-alt',
        color: 'scan',
        description: 'Security vulnerability scan'
    },
    'Approve': {
        icon: 'fa-check-circle',
        color: 'approve',
        description: 'Manual approval gate'
    },
    'Deploy': {
        icon: 'fa-rocket',
        color: 'deploy',
        description: 'Deploy to production'
    },
    'Monitor': {
        icon: 'fa-chart-line',
        color: 'monitor',
        description: 'Monitor application'
    },
    'Rollback': {
        icon: 'fa-undo',
        color: 'rollback',
        description: 'Rollback if needed'
    }
};

// Initialize Game
function initGame() {
    // Show instructions modal on first visit
    const hasPlayedBefore = localStorage.getItem('cicd-game-played');
    if (!hasPlayedBefore) {
        showModal();
    }
    
    setupEventListeners();
    loadLevel(1);
}

// Setup Event Listeners
function setupEventListeners() {
    // Modal controls
    document.getElementById('show-instructions').addEventListener('click', showModal);
    document.getElementById('start-game-btn').addEventListener('click', hideModal);
    document.querySelector('.modal-close').addEventListener('click', hideModal);
    
    // Game controls
    document.getElementById('validate-pipeline').addEventListener('click', validatePipeline);
    document.getElementById('reset-game').addEventListener('click', resetLevel);
    document.getElementById('show-hint').addEventListener('click', showHint);
    document.getElementById('next-level-btn').addEventListener('click', nextLevel);
    document.getElementById('retry-btn').addEventListener('click', retryLevel);
    
    // Click outside modal to close
    document.getElementById('instructions-modal').addEventListener('click', (e) => {
        if (e.target.id === 'instructions-modal') {
            hideModal();
        }
    });
}

// Load Level
function loadLevel(levelNum) {
    if (levelNum > levels.length) {
        showCompletionMessage();
        return;
    }
    
    gameState.currentLevel = levelNum;
    const level = levels[levelNum - 1];
    
    // Update UI
    document.getElementById('current-level').textContent = levelNum;
    document.getElementById('level-title').textContent = level.title;
    document.getElementById('level-description').textContent = level.description;
    
    // Reset pipeline
    gameState.pipelineStages = [];
    renderPipelineSlots(level.requiredStages.length);
    renderAvailableStages(level.availableStages);
    
    // Hide validation result
    document.getElementById('validation-result').classList.add('hidden');
    document.getElementById('hint-box').classList.add('hidden');
    
    // Start timer
    startTimer();
}

// Render Pipeline Slots
function renderPipelineSlots(count) {
    const slotsContainer = document.getElementById('pipeline-slots');
    slotsContainer.innerHTML = '';
    
    for (let i = 0; i < count; i++) {
        const slot = document.createElement('div');
        slot.className = 'pipeline-slot';
        slot.dataset.position = `Step ${i + 1}`;
        slot.dataset.index = i;
        
        // Add drop event listeners
        slot.addEventListener('dragover', handleDragOver);
        slot.addEventListener('drop', handleDrop);
        slot.addEventListener('dragleave', handleDragLeave);
        
        slotsContainer.appendChild(slot);
    }
    
    // Hide instruction if slots are rendered
    document.querySelector('.pipeline-instruction').style.display = 'none';
}

// Render Available Stages
function renderAvailableStages(stages) {
    const container = document.getElementById('available-stages');
    container.innerHTML = '';
    
    stages.forEach(stageName => {
        const stage = createStageElement(stageName);
        container.appendChild(stage);
    });
}

// Create Stage Element
function createStageElement(stageName) {
    const stageDef = stageDefinitions[stageName];
    const stage = document.createElement('div');
    stage.className = 'pipeline-stage';
    stage.draggable = true;
    stage.dataset.stage = stageName;
    
    stage.innerHTML = `
        <div class="stage-icon ${stageDef.color}">
            <i class="fas ${stageDef.icon}"></i>
        </div>
        <div class="stage-details">
            <span class="stage-name">${stageName}</span>
            <span class="stage-description">${stageDef.description}</span>
        </div>
    `;
    
    // Add drag event listeners
    stage.addEventListener('dragstart', handleDragStart);
    stage.addEventListener('dragend', handleDragEnd);
    
    // Touch support for mobile
    stage.addEventListener('touchstart', handleTouchStart, { passive: false });
    
    return stage;
}

// Drag and Drop Handlers
let draggedElement = null;

function handleDragStart(e) {
    draggedElement = e.target.closest('.pipeline-stage');
    draggedElement.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/html', draggedElement.innerHTML);
}

function handleDragEnd(e) {
    if (draggedElement) {
        draggedElement.classList.remove('dragging');
    }
}

function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    const slot = e.currentTarget;
    if (!slot.classList.contains('filled')) {
        slot.classList.add('drag-over');
    }
    return false;
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    
    const slot = e.currentTarget;
    slot.classList.remove('drag-over');
    
    if (!draggedElement || slot.classList.contains('filled')) {
        return;
    }
    
    // Get stage name
    const stageName = draggedElement.dataset.stage;
    const slotIndex = parseInt(slot.dataset.index);
    
    // Add stage to pipeline
    const stageClone = createStageElement(stageName);
    stageClone.draggable = false;
    stageClone.style.cursor = 'default';
    
    // Add remove button
    const removeBtn = document.createElement('button');
    removeBtn.innerHTML = '<i class="fas fa-times"></i>';
    removeBtn.className = 'remove-stage-btn';
    removeBtn.style.cssText = `
        position: absolute;
        top: 5px;
        right: 5px;
        background: #f44336;
        color: white;
        border: none;
        border-radius: 50%;
        width: 25px;
        height: 25px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
    `;
    removeBtn.onclick = () => removeStageFromSlot(slotIndex);
    stageClone.appendChild(removeBtn);
    
    slot.innerHTML = '';
    slot.appendChild(stageClone);
    slot.classList.add('filled');
    
    // Update game state
    gameState.pipelineStages[slotIndex] = stageName;
    
    // Mark original as used
    draggedElement.classList.add('used');
    
    return false;
}

// Touch Support for Mobile
function handleTouchStart(e) {
    const stage = e.target.closest('.pipeline-stage');
    if (!stage || stage.classList.contains('used')) return;
    
    e.preventDefault();
    
    // Show available slots
    const slots = document.querySelectorAll('.pipeline-slot:not(.filled)');
    if (slots.length === 0) return;
    
    // Create touch selection
    const stageName = stage.dataset.stage;
    const confirmMsg = `Add "${stageName}" to next available slot?`;
    
    if (confirm(confirmMsg)) {
        const emptySlot = slots[0];
        const slotIndex = parseInt(emptySlot.dataset.index);
        
        // Add stage
        const stageClone = createStageElement(stageName);
        stageClone.draggable = false;
        
        const removeBtn = document.createElement('button');
        removeBtn.innerHTML = '<i class="fas fa-times"></i>';
        removeBtn.className = 'remove-stage-btn';
        removeBtn.style.cssText = `
            position: absolute;
            top: 5px;
            right: 5px;
            background: #f44336;
            color: white;
            border: none;
            border-radius: 50%;
            width: 25px;
            height: 25px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        `;
        removeBtn.onclick = () => removeStageFromSlot(slotIndex);
        stageClone.appendChild(removeBtn);
        
        emptySlot.innerHTML = '';
        emptySlot.appendChild(stageClone);
        emptySlot.classList.add('filled');
        
        gameState.pipelineStages[slotIndex] = stageName;
        stage.classList.add('used');
    }
}

// Remove Stage from Slot
function removeStageFromSlot(index) {
    const slots = document.querySelectorAll('.pipeline-slot');
    const slot = slots[index];
    
    const stageName = gameState.pipelineStages[index];
    
    // Clear slot
    slot.innerHTML = '';
    slot.classList.remove('filled');
    
    // Remove from game state
    gameState.pipelineStages[index] = null;
    
    // Re-enable stage in toolbox
    const availableStages = document.querySelectorAll('.pipeline-stage');
    availableStages.forEach(stage => {
        if (stage.dataset.stage === stageName) {
            stage.classList.remove('used');
        }
    });
}

// Validate Pipeline
function validatePipeline() {
    const level = levels[gameState.currentLevel - 1];
    const requiredStages = level.requiredStages;
    
    // Check if all slots are filled
    const filledCount = gameState.pipelineStages.filter(s => s !== null && s !== undefined).length;
    
    if (filledCount < requiredStages.length) {
        showValidationResult(false, 'Incomplete Pipeline!', 
            `You need to fill all ${requiredStages.length} stages. Currently filled: ${filledCount}`);
        return;
    }
    
    // Check if pipeline matches required order
    let isCorrect = true;
    for (let i = 0; i < requiredStages.length; i++) {
        if (gameState.pipelineStages[i] !== requiredStages[i]) {
            isCorrect = false;
            break;
        }
    }
    
    if (isCorrect) {
        // Calculate score based on time
        const timeBonus = Math.max(0, 100 - Math.floor(getElapsedTime() / 10));
        const levelScore = 100 + timeBonus;
        gameState.score += levelScore;
        
        document.getElementById('game-score').textContent = gameState.score;
        
        showValidationResult(true, '🎉 Pipeline Deployed!', 
            `Perfect! Your CI/CD pipeline is correctly configured. Score: +${levelScore}`);
        
        // Mark as played
        localStorage.setItem('cicd-game-played', 'true');
    } else {
        showValidationResult(false, '❌ Pipeline Failed!', 
            'The stages are not in the correct order. Check the hint or try again!');
    }
}

// Show Validation Result
function showValidationResult(success, title, message) {
    const resultDiv = document.getElementById('validation-result');
    const icon = resultDiv.querySelector('.result-icon');
    const titleEl = resultDiv.querySelector('.result-title');
    const messageEl = resultDiv.querySelector('.result-message');
    
    icon.className = success ? 'result-icon success' : 'result-icon error';
    titleEl.textContent = title;
    messageEl.textContent = message;
    
    // Show appropriate buttons
    const nextBtn = document.getElementById('next-level-btn');
    const retryBtn = document.getElementById('retry-btn');
    
    if (success && gameState.currentLevel < levels.length) {
        nextBtn.style.display = 'inline-flex';
        retryBtn.style.display = 'none';
    } else if (success && gameState.currentLevel === levels.length) {
        nextBtn.textContent = '🏆 View Results';
        nextBtn.style.display = 'inline-flex';
        retryBtn.style.display = 'none';
    } else {
        nextBtn.style.display = 'none';
        retryBtn.style.display = 'inline-flex';
    }
    
    resultDiv.classList.remove('hidden');
}

// Show Hint
function showHint() {
    const level = levels[gameState.currentLevel - 1];
    const hintBox = document.getElementById('hint-box');
    const hintText = document.getElementById('hint-text');
    
    hintText.textContent = level.hint;
    hintBox.classList.remove('hidden');
    
    // Auto-hide after 10 seconds
    setTimeout(() => {
        hintBox.classList.add('hidden');
    }, 10000);
}

// Next Level
function nextLevel() {
    if (gameState.currentLevel >= levels.length) {
        showCompletionMessage();
    } else {
        loadLevel(gameState.currentLevel + 1);
    }
}

// Retry Level
function retryLevel() {
    document.getElementById('validation-result').classList.add('hidden');
    resetLevel();
}

// Reset Level
function resetLevel() {
    loadLevel(gameState.currentLevel);
}

// Timer Functions
function startTimer() {
    gameState.startTime = Date.now();
    
    if (gameState.timer) {
        clearInterval(gameState.timer);
    }
    
    gameState.timer = setInterval(() => {
        const elapsed = getElapsedTime();
        document.getElementById('game-time').textContent = elapsed + 's';
    }, 1000);
}

function getElapsedTime() {
    return Math.floor((Date.now() - gameState.startTime) / 1000);
}

// Show Completion Message
function showCompletionMessage() {
    const totalTime = getElapsedTime();
    showValidationResult(true, '🏆 All Levels Complete!', 
        `Congratulations! You've mastered CI/CD pipeline design! Final Score: ${gameState.score} | Time: ${totalTime}s`);
    
    if (gameState.timer) {
        clearInterval(gameState.timer);
    }
}

// Modal Functions
function showModal() {
    document.getElementById('instructions-modal').classList.remove('hidden');
}

function hideModal() {
    document.getElementById('instructions-modal').classList.add('hidden');
}

// Initialize game when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initGame);
} else {
    initGame();
}

console.log('🎮 CI/CD Pipeline Builder Game loaded!');
