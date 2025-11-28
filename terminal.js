// ================================================
// INTERACTIVE TERMINAL
// ================================================

const terminalData = {
    name: "Rakesh Kumar Mallam",
    role: "Senior Technical Consultant",
    company: "Red Hat",
    location: "Australia",
    experience: "15+ years",
    email: "mallamrakesh@gmail.com",
    linkedin: "linkedin.com/in/rakeshkumarmallam",
    github: "github.com/rmallam",
    
    skills: {
        cloud: ["OpenShift", "Kubernetes", "AWS", "Azure", "Docker", "EKS/AKS"],
        devops: ["Jenkins", "ArgoCD", "Tekton", "Ansible", "Terraform", "GitHub Actions"],
        programming: ["Python", "Go", "Java", "Shell", "React.js"],
        monitoring: ["Prometheus", "Grafana", "Splunk", "AppDynamics"]
    },
    
    certifications: [
        "Red Hat OpenShift Administrator",
        "Red Hat OpenShift Developer", 
        "AWS Solution Architect Associate",
        "Azure Fundamentals"
    ],
    
    achievements: [
        "🏆 Top Gun Consultant - Red Hat ANZ",
        "⭐ Most Trusted Consultant - Red Hat ANZ",
        "📈 High Performer - Every Review at NAB",
        "🌟 S*tar Rating - 4 years at Tech Mahindra"
    ]
};

const commands = {
    help: () => {
        return `
<span class="terminal-success">Available Commands:</span>

  <span class="cmd-highlight">whoami</span>      - Learn about me
  <span class="cmd-highlight">skills</span>      - View my technical skills
  <span class="cmd-highlight">experience</span>  - See my work history
  <span class="cmd-highlight">certs</span>       - List certifications
  <span class="cmd-highlight">achievements</span> - View my achievements
  <span class="cmd-highlight">contact</span>     - Get my contact info
  <span class="cmd-highlight">download</span>    - Download my resume
  <span class="cmd-highlight">social</span>      - Social media links
  <span class="cmd-highlight">hire</span>        - Why you should hire me
  <span class="cmd-highlight">clear</span>       - Clear the terminal
  <span class="cmd-highlight">exit</span>        - Close terminal`;
    },
    
    whoami: () => {
        return `
<span class="terminal-success">👋 Hello! I'm ${terminalData.name}</span>

Role: ${terminalData.role}
Company: ${terminalData.company}
Location: ${terminalData.location}
Experience: ${terminalData.experience}

I'm a passionate cloud architect and DevOps expert who loves 
building scalable, secure infrastructure. When I'm not coding,
you'll find me traveling with family or exploring new cuisines!`;
    },
    
    skills: () => {
        let output = `<span class="terminal-success">🛠️ Technical Skills:</span>\n\n`;
        output += `<span class="cmd-highlight">Cloud & Containers:</span>\n  ${terminalData.skills.cloud.join(", ")}\n\n`;
        output += `<span class="cmd-highlight">DevOps & Automation:</span>\n  ${terminalData.skills.devops.join(", ")}\n\n`;
        output += `<span class="cmd-highlight">Programming:</span>\n  ${terminalData.skills.programming.join(", ")}\n\n`;
        output += `<span class="cmd-highlight">Monitoring:</span>\n  ${terminalData.skills.monitoring.join(", ")}`;
        return output;
    },
    
    experience: () => {
        return `
<span class="terminal-success">💼 Work Experience:</span>

<span class="cmd-highlight">Red Hat</span> (2020 - Present)
  Sr Architect / Consultant
  → Trusted Advisor for enterprise cloud transformations
  → OpenShift, Azure ARO, AWS architectures

<span class="cmd-highlight">National Australia Bank</span> (2016 - 2020)
  Technical Lead → Sr Engineer
  → Enterprise DevOps platform on AWS
  → Blue/Green deployments, CI/CD pipelines

<span class="cmd-highlight">Infosys / Tech Mahindra</span> (2010 - 2016)
  DevOps Lead → DevOps Engineer
  → Ansible automation, Jenkins pipelines
  → WebSphere, AWS infrastructure`;
    },
    
    certs: () => {
        let output = `<span class="terminal-success">🎓 Certifications:</span>\n\n`;
        terminalData.certifications.forEach(cert => {
            output += `  ✓ ${cert}\n`;
        });
        return output;
    },
    
    achievements: () => {
        let output = `<span class="terminal-success">🏅 Achievements & Recognition:</span>\n\n`;
        terminalData.achievements.forEach(achievement => {
            output += `  ${achievement}\n`;
        });
        return output;
    },
    
    contact: () => {
        return `
<span class="terminal-success">📬 Contact Information:</span>

  📧 Email: ${terminalData.email}
  🔗 LinkedIn: ${terminalData.linkedin}
  💻 GitHub: ${terminalData.github}

Type <span class="cmd-highlight">social</span> to open links directly!`;
    },
    
    social: () => {
        return `
<span class="terminal-success">🌐 Social Links:</span>

  Opening LinkedIn... <a href="https://${terminalData.linkedin}" target="_blank" style="color: #58A6FF;">Click here</a>
  Opening GitHub... <a href="https://${terminalData.github}" target="_blank" style="color: #58A6FF;">Click here</a>`;
    },
    
    download: () => {
        // Trigger resume download
        const link = document.querySelector('.resume-float-btn');
        if (link) link.click();
        return `<span class="terminal-success">📄 Downloading resume...</span>\n\nIf download doesn't start, click the Resume button on the right.`;
    },
    
    hire: () => {
        return `
<span class="terminal-success">💡 Why Hire Me?</span>

  ✅ 15+ years of enterprise-scale experience
  ✅ Proven track record at Red Hat, NAB, Infosys
  ✅ Expert in Cloud Native & DevSecOps
  ✅ Award-winning consultant (Top Gun, Most Trusted)
  ✅ Strong communicator & team leader
  ✅ Passionate about continuous learning

<span class="cmd-highlight">Ready to transform your infrastructure?</span>
Type <span class="cmd-highlight">contact</span> to get in touch!`;
    },
    
    clear: () => {
        return '__CLEAR__';
    },
    
    exit: () => {
        return '__EXIT__';
    },
    
    ls: () => {
        return `about.txt  skills.json  experience.md  certs/  projects/  contact.txt`;
    },
    
    pwd: () => {
        return `/home/visitor/rakesh-portfolio`;
    },
    
    date: () => {
        return new Date().toString();
    },
    
    echo: (args) => {
        return args || '';
    },
    
    cat: (args) => {
        if (args === 'about.txt') return commands.whoami();
        if (args === 'skills.json') return commands.skills();
        if (args === 'contact.txt') return commands.contact();
        return `<span class="terminal-error">cat: ${args || 'file'}: No such file</span>`;
    },
    
    sudo: () => {
        return `<span class="terminal-error">Nice try! 😄 But you don't have sudo access here.</span>\nType <span class="cmd-highlight">help</span> to see what you can do.`;
    },
    
    rm: () => {
        return `<span class="terminal-error">🚫 Permission denied: Can't delete my portfolio!</span>`;
    },
    
    vim: () => {
        return `<span class="terminal-success">Good choice! But this terminal doesn't have vim.</span>\nType <span class="cmd-highlight">skills</span> to see that I know vim though! 😉`;
    },
    
    coffee: () => {
        return `<span class="terminal-success">☕ Brewing coffee...</span>\n\nHere's your virtual coffee! ☕\nI'm always up for a chat over coffee. Type <span class="cmd-highlight">contact</span>!`;
    },
    
    hello: () => {
        return `<span class="terminal-success">👋 Hello there!</span>\n\nWelcome to my portfolio! Type <span class="cmd-highlight">help</span> to explore.`;
    },
    
    hi: () => commands.hello(),
    
    joke: () => {
        const jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
            "A SQL query walks into a bar, walks up to two tables and asks... 'Can I join you?'",
            "Why do DevOps engineers make terrible comedians? Because they always automate the punchline!",
            "There are only 10 types of people: those who understand binary and those who don't.",
            "Why did the Kubernetes pod go to therapy? It had too many container issues!"
        ];
        return `<span class="terminal-success">😄 Dev Joke:</span>\n\n${jokes[Math.floor(Math.random() * jokes.length)]}`;
    }
};

// Terminal functionality
let terminalHistory = [];
let historyIndex = -1;

function initTerminal() {
    const terminalToggle = document.getElementById('terminal-toggle');
    const terminalContainer = document.getElementById('terminal-container');
    const terminalClose = document.getElementById('terminal-close');
    const terminalInput = document.getElementById('terminal-input');
    const terminalOutput = document.getElementById('terminal-output');
    const closeBtn = document.querySelector('.term-btn.close');
    
    if (!terminalToggle || !terminalContainer) return;
    
    // Toggle terminal
    terminalToggle.addEventListener('click', () => {
        terminalContainer.classList.toggle('hidden');
        if (!terminalContainer.classList.contains('hidden')) {
            terminalInput.focus();
        }
    });
    
    // Close terminal
    terminalClose.addEventListener('click', () => {
        terminalContainer.classList.add('hidden');
    });
    
    closeBtn.addEventListener('click', () => {
        terminalContainer.classList.add('hidden');
    });
    
    // Handle input
    terminalInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            const input = terminalInput.value.trim();
            if (input) {
                processCommand(input);
                terminalHistory.push(input);
                historyIndex = terminalHistory.length;
            }
            terminalInput.value = '';
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            if (historyIndex > 0) {
                historyIndex--;
                terminalInput.value = terminalHistory[historyIndex];
            }
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (historyIndex < terminalHistory.length - 1) {
                historyIndex++;
                terminalInput.value = terminalHistory[historyIndex];
            } else {
                historyIndex = terminalHistory.length;
                terminalInput.value = '';
            }
        } else if (e.key === 'Escape') {
            terminalContainer.classList.add('hidden');
        }
    });
    
    // Click terminal body to focus input
    document.querySelector('.terminal-body').addEventListener('click', () => {
        terminalInput.focus();
    });
}

function processCommand(input) {
    const terminalOutput = document.getElementById('terminal-output');
    const parts = input.toLowerCase().split(' ');
    const cmd = parts[0];
    const args = parts.slice(1).join(' ');
    
    // Add command to output
    const commandLine = document.createElement('p');
    commandLine.className = 'terminal-command';
    commandLine.innerHTML = `<span class="terminal-prompt">visitor@rakesh.dev:~$</span> ${escapeHtml(input)}`;
    terminalOutput.appendChild(commandLine);
    
    // Process command
    let response;
    if (commands[cmd]) {
        response = typeof commands[cmd] === 'function' ? commands[cmd](args) : commands[cmd];
    } else {
        response = `<span class="terminal-error">Command not found: ${escapeHtml(cmd)}</span>\nType <span class="cmd-highlight">help</span> for available commands.`;
    }
    
    // Handle special responses
    if (response === '__CLEAR__') {
        terminalOutput.innerHTML = `
            <p class="terminal-welcome">Terminal cleared! 🧹</p>
            <p class="terminal-hint">Type <span class="cmd-highlight">help</span> to see available commands.</p>
        `;
        return;
    }
    
    if (response === '__EXIT__') {
        const container = document.getElementById('terminal-container');
        container.classList.add('hidden');
        return;
    }
    
    // Add response
    const responseLine = document.createElement('div');
    responseLine.className = 'terminal-response';
    responseLine.innerHTML = response.replace(/\n/g, '<br>');
    terminalOutput.appendChild(responseLine);
    
    // Scroll to bottom
    terminalOutput.scrollTop = terminalOutput.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTerminal);
} else {
    initTerminal();
}

console.log('🖥️ Terminal loaded! Click the terminal button to open.');

