// ================================================
// DATA LOADER - Loads content from data.json
// ================================================

const DATA_URL = 'https://raw.githubusercontent.com/rmallam/rmallam.github.io/main/data.json';

let siteData = null;

// Load data and populate the website
async function loadSiteData() {
    try {
        const response = await fetch(DATA_URL);
        if (!response.ok) throw new Error('Failed to load data');
        siteData = await response.json();
        
        populatePersonalInfo();
        populateAbout();
        populateExperience();
        populateSkills();
        populateCertifications();
        populateRecognition();
        populateContact();
        updateTerminalData();
        
        console.log('✅ Site data loaded successfully!');
    } catch (error) {
        console.error('Error loading site data:', error);
        // Site will show default HTML content if data fails to load
    }
}

// Populate personal info (hero section)
function populatePersonalInfo() {
    const { personal, contact } = siteData;
    
    // Hero section
    const heroTitle = document.querySelector('.hero-title');
    if (heroTitle) {
        const nameParts = personal.name.split(' ');
        const lastName = nameParts.pop();
        const firstName = nameParts.join(' ');
        heroTitle.innerHTML = `${firstName} <span class="highlight">${lastName}</span>`;
    }
    
    const heroRole = document.querySelector('.hero-role');
    if (heroRole) heroRole.textContent = personal.role;
    
    const heroSubtitle = document.querySelector('.hero-subtitle');
    if (heroSubtitle) heroSubtitle.textContent = personal.tagline;
    
    const heroDescription = document.querySelector('.hero-description');
    if (heroDescription) heroDescription.textContent = personal.description;
    
    const heroBadge = document.querySelector('.hero-badge');
    if (heroBadge) {
        heroBadge.innerHTML = `<i class="fas fa-coffee"></i> ${personal.availability}`;
    }
    
    // Resume button
    const resumeBtn = document.getElementById('resume-btn');
    if (resumeBtn && personal.resumeFile) {
        resumeBtn.href = personal.resumeFile;
    }
    
    // GitHub button
    const githubBtn = document.querySelector('.hero-buttons .btn-secondary');
    if (githubBtn && contact.github) {
        githubBtn.href = contact.github;
    }
}

// Populate about section
function populateAbout() {
    const { about } = siteData;
    
    const aboutText = document.querySelector('.about-text');
    if (aboutText) {
        aboutText.innerHTML = `
            <p>${about.intro}</p>
            <p>${about.expertise}</p>
            <p>${about.personal.replace('blog', '<a href="#blogs">blog</a>')}</p>
        `;
    }
    
    // Stats
    const statCards = document.querySelectorAll('.stat-card');
    if (statCards.length >= 3) {
        statCards[0].querySelector('h3').textContent = about.stats.yearsExperience;
        statCards[1].querySelector('h3').textContent = about.stats.projectsDelivered;
        statCards[2].querySelector('h3').textContent = about.stats.certifications;
    }
}

// Populate experience section
function populateExperience() {
    const { experience } = siteData;
    
    const timeline = document.querySelector('.timeline');
    if (!timeline) return;
    
    timeline.innerHTML = experience.map(job => `
        <div class="timeline-item">
            <div class="timeline-marker"></div>
            <div class="timeline-content">
                <div class="timeline-header">
                    <h3>${job.title}</h3>
                    <span class="company"><i class="${job.icon}"></i> ${job.company}</span>
                    <span class="date">${job.period}</span>
                </div>
                <ul class="timeline-list">
                    ${job.highlights.map(h => `<li>${h}</li>`).join('')}
                </ul>
                <div class="tech-tags">
                    ${job.technologies.map(t => `<span class="tag">${t}</span>`).join('')}
                </div>
            </div>
        </div>
    `).join('');
}

// Populate skills section
function populateSkills() {
    const { skills } = siteData;
    
    const skillsGrid = document.querySelector('.skills-grid');
    if (!skillsGrid) return;
    
    skillsGrid.innerHTML = skills.map(category => `
        <div class="skill-category">
            <h3><i class="${category.icon}"></i> ${category.category}</h3>
            <div class="skill-items">
                ${category.items.map(item => `<span class="skill-item">${item}</span>`).join('')}
            </div>
        </div>
    `).join('');
}

// Populate certifications section
function populateCertifications() {
    const { certifications } = siteData;
    
    const certGrid = document.querySelector('.cert-grid');
    if (!certGrid) return;
    
    certGrid.innerHTML = certifications.map(cert => `
        <div class="cert-card">
            <div class="cert-icon ${cert.color}">
                <i class="${cert.icon}"></i>
            </div>
            <h3>${cert.name}</h3>
            <p class="cert-issuer">${cert.issuer}</p>
        </div>
    `).join('');
}

// Populate recognition section
function populateRecognition() {
    const { recognition } = siteData;
    
    const recognitionGrid = document.querySelector('.recognition-grid');
    if (!recognitionGrid) return;
    
    recognitionGrid.innerHTML = recognition.map(item => `
        <div class="recognition-item">
            <i class="${item.icon}"></i>
            <p><strong>${item.title}</strong>${item.description}</p>
        </div>
    `).join('');
}

// Populate contact section
function populateContact() {
    const { contact } = siteData;
    
    const contactMethods = document.querySelector('.contact-methods');
    if (!contactMethods) return;
    
    contactMethods.innerHTML = `
        <a href="tel:${contact.phone}" class="contact-card">
            <i class="fas fa-phone"></i>
            <h3>Phone</h3>
            <p>${contact.phone}</p>
        </a>
        <a href="mailto:${contact.email}" class="contact-card">
            <i class="fas fa-envelope"></i>
            <h3>Email</h3>
            <p>${contact.email}</p>
        </a>
        <a href="${contact.linkedin}" target="_blank" rel="noopener" class="contact-card">
            <i class="fab fa-linkedin"></i>
            <h3>LinkedIn</h3>
            <p>Connect with me</p>
        </a>
        <a href="${contact.github}" target="_blank" rel="noopener" class="contact-card">
            <i class="fab fa-github"></i>
            <h3>GitHub</h3>
            <p>View my projects</p>
        </a>
    `;
    
    // Update footer links
    const footerGithub = document.querySelector('.footer-links a[aria-label="GitHub"]');
    const footerLinkedin = document.querySelector('.footer-links a[aria-label="LinkedIn"]');
    if (footerGithub) footerGithub.href = contact.github;
    if (footerLinkedin) footerLinkedin.href = contact.linkedin;
}

// Update terminal data
function updateTerminalData() {
    if (typeof terminalData === 'undefined') return;
    
    const { personal, contact, skills, certifications, recognition } = siteData;
    
    // Update terminal data object
    terminalData.name = personal.name;
    terminalData.role = personal.role;
    terminalData.company = personal.company;
    terminalData.location = personal.location;
    terminalData.email = contact.email;
    terminalData.linkedin = contact.linkedin.replace('https://', '');
    terminalData.github = contact.github.replace('https://', '');
    
    // Update skills
    terminalData.skills = {
        cloud: skills.find(s => s.category.includes('Cloud'))?.items || [],
        devops: skills.find(s => s.category.includes('DevOps'))?.items || [],
        programming: skills.find(s => s.category.includes('Programming'))?.items || [],
        monitoring: skills.find(s => s.category.includes('Monitoring'))?.items || []
    };
    
    // Update certifications
    terminalData.certifications = certifications.map(c => c.name);
    
    // Update achievements
    terminalData.achievements = recognition.map(r => `${r.title} - ${r.description}`);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadSiteData);
} else {
    loadSiteData();
}

console.log('📦 Data loader initialized');

