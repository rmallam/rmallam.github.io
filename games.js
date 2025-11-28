// ================================================
// GAME CAROUSEL & ADDITIONAL GAMES
// ================================================

// Game Carousel Navigation
function initGameCarousel() {
    const tabs = document.querySelectorAll('.game-tab');
    const slides = document.querySelectorAll('.game-slide');
    const prevBtn = document.getElementById('game-prev');
    const nextBtn = document.getElementById('game-next');
    let currentGame = 0;

    function showGame(index) {
        // Update tabs
        tabs.forEach(tab => tab.classList.remove('active'));
        tabs[index].classList.add('active');

        // Update slides
        slides.forEach(slide => slide.classList.remove('active'));
        slides[index].classList.add('active');

        currentGame = index;
    }

    // Tab clicks
    tabs.forEach((tab, index) => {
        tab.addEventListener('click', () => showGame(index));
    });

    // Arrow navigation
    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            const newIndex = currentGame === 0 ? slides.length - 1 : currentGame - 1;
            showGame(newIndex);
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            const newIndex = currentGame === slides.length - 1 ? 0 : currentGame + 1;
            showGame(newIndex);
        });
    }

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT') return;
        if (e.key === 'ArrowLeft' && prevBtn) prevBtn.click();
        if (e.key === 'ArrowRight' && nextBtn) nextBtn.click();
    });
}

// ================================================
// CLOUD QUIZ GAME
// ================================================
const quizQuestions = [
    {
        question: "What does CI/CD stand for?",
        options: [
            "Continuous Integration / Continuous Deployment",
            "Computer Integration / Computer Deployment",
            "Code Integration / Code Deployment",
            "Container Integration / Container Deployment"
        ],
        correct: 0
    },
    {
        question: "Which tool is used for container orchestration?",
        options: ["Jenkins", "Kubernetes", "Ansible", "Terraform"],
        correct: 1
    },
    {
        question: "What is OpenShift built on top of?",
        options: ["Docker Swarm", "Kubernetes", "Mesos", "Nomad"],
        correct: 1
    },
    {
        question: "Which cloud provider offers 'Lambda' as a serverless compute service?",
        options: ["Google Cloud", "Microsoft Azure", "Amazon Web Services", "IBM Cloud"],
        correct: 2
    },
    {
        question: "What does IaC stand for in DevOps?",
        options: [
            "Internet as Code",
            "Infrastructure as Code",
            "Integration as Code",
            "Iteration as Code"
        ],
        correct: 1
    },
    {
        question: "Which tool is commonly used for GitOps deployments?",
        options: ["Jenkins", "ArgoCD", "Maven", "Gradle"],
        correct: 1
    },
    {
        question: "What is the primary purpose of a Service Mesh?",
        options: [
            "Database management",
            "Service-to-service communication",
            "File storage",
            "User authentication"
        ],
        correct: 1
    },
    {
        question: "Which command lists all pods in Kubernetes?",
        options: [
            "kubectl list pods",
            "kubectl show pods",
            "kubectl get pods",
            "kubectl pods list"
        ],
        correct: 2
    },
    {
        question: "What is Terraform primarily used for?",
        options: [
            "Container runtime",
            "Configuration management",
            "Infrastructure provisioning",
            "Log aggregation"
        ],
        correct: 2
    },
    {
        question: "Which protocol does Prometheus use to collect metrics?",
        options: ["Push-based", "Pull-based (HTTP scraping)", "SNMP", "MQTT"],
        correct: 1
    }
];

let quizState = {
    currentQuestion: 0,
    score: 0,
    timeLeft: 30,
    timer: null,
    isActive: false
};

function initQuizGame() {
    const startBtn = document.getElementById('start-quiz');
    const resetBtn = document.getElementById('reset-quiz');
    const nextBtn = document.getElementById('next-question');

    if (startBtn) {
        startBtn.addEventListener('click', startQuiz);
    }
    if (resetBtn) {
        resetBtn.addEventListener('click', resetQuiz);
    }
    if (nextBtn) {
        nextBtn.addEventListener('click', nextQuestion);
    }
}

function startQuiz() {
    quizState = {
        currentQuestion: 0,
        score: 0,
        timeLeft: 30,
        timer: null,
        isActive: true
    };

    document.getElementById('start-quiz').classList.add('hidden');
    document.getElementById('quiz-score').textContent = '0';
    showQuestion();
    startQuizTimer();
}

function showQuestion() {
    const q = quizQuestions[quizState.currentQuestion];
    document.getElementById('quiz-question-num').textContent = quizState.currentQuestion + 1;
    document.getElementById('quiz-question').innerHTML = `<h3>${q.question}</h3>`;

    const optionsContainer = document.getElementById('quiz-options');
    optionsContainer.innerHTML = q.options.map((opt, i) => `
        <button class="quiz-option" data-option="${i}">${opt}</button>
    `).join('');

    // Add click handlers
    optionsContainer.querySelectorAll('.quiz-option').forEach(btn => {
        btn.addEventListener('click', () => selectAnswer(parseInt(btn.dataset.option)));
    });

    document.getElementById('quiz-feedback').classList.add('hidden');
    document.getElementById('next-question').classList.add('hidden');
}

function selectAnswer(selected) {
    if (!quizState.isActive) return;

    const q = quizQuestions[quizState.currentQuestion];
    const options = document.querySelectorAll('.quiz-option');
    const feedback = document.getElementById('quiz-feedback');

    // Disable all options
    options.forEach(opt => opt.classList.add('disabled'));

    // Mark correct/incorrect
    options[q.correct].classList.add('correct');
    if (selected !== q.correct) {
        options[selected].classList.add('incorrect');
        feedback.className = 'quiz-feedback incorrect';
        feedback.innerHTML = '<i class="fas fa-times-circle"></i><span>Incorrect! The correct answer is highlighted.</span>';
    } else {
        quizState.score += 10;
        document.getElementById('quiz-score').textContent = quizState.score;
        feedback.className = 'quiz-feedback correct';
        feedback.innerHTML = '<i class="fas fa-check-circle"></i><span>Correct! +10 points</span>';
    }

    feedback.classList.remove('hidden');

    // Show next button or finish
    if (quizState.currentQuestion < quizQuestions.length - 1) {
        document.getElementById('next-question').classList.remove('hidden');
    } else {
        endQuiz();
    }
}

function nextQuestion() {
    quizState.currentQuestion++;
    quizState.timeLeft = 30;
    showQuestion();
}

function startQuizTimer() {
    quizState.timer = setInterval(() => {
        quizState.timeLeft--;
        document.getElementById('quiz-time').textContent = quizState.timeLeft + 's';

        if (quizState.timeLeft <= 0) {
            // Auto-select wrong answer (timeout)
            selectAnswer(-1);
        }
    }, 1000);
}

function endQuiz() {
    clearInterval(quizState.timer);
    quizState.isActive = false;

    const feedback = document.getElementById('quiz-feedback');
    feedback.className = 'quiz-feedback correct';
    feedback.innerHTML = `
        <i class="fas fa-trophy"></i>
        <span>Quiz Complete! Final Score: ${quizState.score}/${quizQuestions.length * 10}</span>
    `;
    feedback.classList.remove('hidden');

    document.getElementById('next-question').classList.add('hidden');
    document.getElementById('start-quiz').textContent = 'Play Again';
    document.getElementById('start-quiz').innerHTML = '<i class="fas fa-redo"></i> Play Again';
    document.getElementById('start-quiz').classList.remove('hidden');
}

function resetQuiz() {
    clearInterval(quizState.timer);
    quizState = {
        currentQuestion: 0,
        score: 0,
        timeLeft: 30,
        timer: null,
        isActive: false
    };

    document.getElementById('quiz-score').textContent = '0';
    document.getElementById('quiz-time').textContent = '30s';
    document.getElementById('quiz-question-num').textContent = '1';
    document.getElementById('quiz-question').innerHTML = '<h3>Click "Start Quiz" to begin!</h3>';
    document.getElementById('quiz-options').innerHTML = '';
    document.getElementById('quiz-feedback').classList.add('hidden');
    document.getElementById('next-question').classList.add('hidden');
    document.getElementById('start-quiz').innerHTML = '<i class="fas fa-play"></i> Start Quiz';
    document.getElementById('start-quiz').classList.remove('hidden');
}

// ================================================
// TERMINAL TYPE RACE GAME
// ================================================
const terminalCommands = [
    "kubectl get pods -n production",
    "docker build -t myapp:latest .",
    "terraform init && terraform apply",
    "ansible-playbook deploy.yml",
    "git push origin main --force",
    "helm install redis bitnami/redis",
    "oc new-project my-app",
    "aws s3 sync ./dist s3://bucket",
    "jenkins-cli build my-job",
    "argocd app sync my-app",
    "podman run -d nginx:alpine",
    "tekton pipeline start build",
    "oc adm policy add-role-to-user",
    "kubectl apply -f deployment.yaml",
    "docker-compose up -d --build"
];

let typeState = {
    commands: [],
    currentIndex: 0,
    correctChars: 0,
    totalChars: 0,
    startTime: null,
    timer: null,
    timeLeft: 60,
    isActive: false,
    completedCommands: 0
};

function initTypeRaceGame() {
    const startBtn = document.getElementById('start-typerace');
    const resetBtn = document.getElementById('reset-typerace');
    const input = document.getElementById('type-input');

    if (startBtn) {
        startBtn.addEventListener('click', startTypeRace);
    }
    if (resetBtn) {
        resetBtn.addEventListener('click', resetTypeRace);
    }
    if (input) {
        input.addEventListener('input', handleTypeInput);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && typeState.isActive) {
                checkCommand();
            }
        });
    }
}

function startTypeRace() {
    // Shuffle and pick 10 commands
    typeState.commands = [...terminalCommands].sort(() => Math.random() - 0.5).slice(0, 10);
    typeState.currentIndex = 0;
    typeState.correctChars = 0;
    typeState.totalChars = 0;
    typeState.startTime = Date.now();
    typeState.timeLeft = 60;
    typeState.isActive = true;
    typeState.completedCommands = 0;

    document.getElementById('start-typerace').classList.add('hidden');
    document.getElementById('type-results').classList.add('hidden');
    document.getElementById('type-input').disabled = false;
    document.getElementById('type-input').focus();

    showCommand();
    startTypeTimer();
}

function showCommand() {
    const cmd = typeState.commands[typeState.currentIndex];
    document.getElementById('type-command').innerHTML = `<code>${cmd}</code>`;
    document.getElementById('type-input').value = '';
    document.getElementById('type-commands').textContent = typeState.completedCommands;
    updateTypeFeedback();
}

function handleTypeInput() {
    if (!typeState.isActive) return;

    const input = document.getElementById('type-input').value;
    const target = typeState.commands[typeState.currentIndex];

    // Highlight matching characters
    let html = '';
    for (let i = 0; i < target.length; i++) {
        if (i < input.length) {
            if (input[i] === target[i]) {
                html += `<span style="color: #22C55E">${target[i]}</span>`;
            } else {
                html += `<span style="color: #EF4444; text-decoration: underline">${target[i]}</span>`;
            }
        } else {
            html += target[i];
        }
    }
    document.getElementById('type-command').innerHTML = `<code>${html}</code>`;

    updateTypeFeedback();

    // Auto-submit on exact match
    if (input === target) {
        checkCommand();
    }
}

function checkCommand() {
    const input = document.getElementById('type-input').value;
    const target = typeState.commands[typeState.currentIndex];

    typeState.totalChars += target.length;

    // Count correct characters
    for (let i = 0; i < Math.min(input.length, target.length); i++) {
        if (input[i] === target[i]) {
            typeState.correctChars++;
        }
    }

    if (input === target) {
        typeState.completedCommands++;
    }

    // Move to next command
    if (typeState.currentIndex < typeState.commands.length - 1) {
        typeState.currentIndex++;
        showCommand();
    } else {
        endTypeRace();
    }
}

function updateTypeFeedback() {
    const input = document.getElementById('type-input').value;
    const target = typeState.commands[typeState.currentIndex];

    let correct = 0;
    for (let i = 0; i < Math.min(input.length, target.length); i++) {
        if (input[i] === target[i]) correct++;
    }

    document.getElementById('type-feedback').innerHTML = `
        <span class="correct-chars" style="color: #22C55E">${correct}</span> / 
        <span class="total-chars">${target.length}</span> characters
    `;
}

function startTypeTimer() {
    typeState.timer = setInterval(() => {
        typeState.timeLeft--;
        document.getElementById('type-time').textContent = typeState.timeLeft + 's';

        // Update WPM
        const elapsed = (Date.now() - typeState.startTime) / 1000 / 60;
        const wpm = elapsed > 0 ? Math.round((typeState.correctChars / 5) / elapsed) : 0;
        document.getElementById('type-wpm').textContent = wpm;

        if (typeState.timeLeft <= 0) {
            endTypeRace();
        }
    }, 1000);
}

function endTypeRace() {
    clearInterval(typeState.timer);
    typeState.isActive = false;

    const elapsed = (Date.now() - typeState.startTime) / 1000 / 60;
    const wpm = elapsed > 0 ? Math.round((typeState.correctChars / 5) / elapsed) : 0;
    const accuracy = typeState.totalChars > 0 
        ? Math.round((typeState.correctChars / typeState.totalChars) * 100) 
        : 0;

    document.getElementById('type-input').disabled = true;
    document.getElementById('final-wpm').textContent = wpm;
    document.getElementById('final-accuracy').textContent = accuracy + '%';
    document.getElementById('final-commands').textContent = typeState.completedCommands;
    document.getElementById('type-results').classList.remove('hidden');

    document.getElementById('start-typerace').innerHTML = '<i class="fas fa-redo"></i> Race Again';
    document.getElementById('start-typerace').classList.remove('hidden');
}

function resetTypeRace() {
    clearInterval(typeState.timer);
    typeState = {
        commands: [],
        currentIndex: 0,
        correctChars: 0,
        totalChars: 0,
        startTime: null,
        timer: null,
        timeLeft: 60,
        isActive: false,
        completedCommands: 0
    };

    document.getElementById('type-time').textContent = '60s';
    document.getElementById('type-wpm').textContent = '0';
    document.getElementById('type-commands').textContent = '0';
    document.getElementById('type-command').innerHTML = '<code>kubectl get pods -n production</code>';
    document.getElementById('type-input').value = '';
    document.getElementById('type-input').disabled = true;
    document.getElementById('type-results').classList.add('hidden');
    document.getElementById('type-feedback').innerHTML = '<span class="correct-chars">0</span> / <span class="total-chars">0</span> characters';
    document.getElementById('start-typerace').innerHTML = '<i class="fas fa-play"></i> Start Race';
    document.getElementById('start-typerace').classList.remove('hidden');
}

// Initialize all games
function initAllGames() {
    initGameCarousel();
    initQuizGame();
    initTypeRaceGame();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAllGames);
} else {
    initAllGames();
}

console.log('🎮 Game Arcade loaded!');

