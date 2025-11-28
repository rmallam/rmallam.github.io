// ================================================
// RAKESH MALLAM PORTFOLIO - MODERN UI SCRIPTS
// ================================================

// DOM Elements
const navbar = document.getElementById('navbar');
const navMenu = document.getElementById('nav-menu');
const navToggle = document.getElementById('nav-toggle');
const navLinks = document.querySelectorAll('.nav-menu a');
const sections = document.querySelectorAll('section[id]');

// ================================================
// MOBILE NAVIGATION
// ================================================
function initMobileNav() {
    if (!navToggle || !navMenu) return;
    
    // Toggle menu on hamburger click
    navToggle.addEventListener('click', () => {
        navMenu.classList.toggle('active');
        
        // Animate hamburger icon
        const icon = navToggle.querySelector('i');
        if (navMenu.classList.contains('active')) {
            icon.classList.remove('fa-bars');
            icon.classList.add('fa-times');
        } else {
            icon.classList.remove('fa-times');
            icon.classList.add('fa-bars');
        }
    });
    
    // Close menu when clicking a link
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            navMenu.classList.remove('active');
            const icon = navToggle.querySelector('i');
            icon.classList.remove('fa-times');
            icon.classList.add('fa-bars');
        });
    });
    
    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!navMenu.contains(e.target) && !navToggle.contains(e.target)) {
            navMenu.classList.remove('active');
            const icon = navToggle.querySelector('i');
            icon.classList.remove('fa-times');
            icon.classList.add('fa-bars');
        }
    });
}

// ================================================
// SMOOTH SCROLLING
// ================================================
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const navbarHeight = 72;
                const additionalPadding = 20;
                const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - navbarHeight - additionalPadding;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// ================================================
// NAVBAR EFFECTS
// ================================================
function updateNavbar() {
    const scrollY = window.pageYOffset;
    
    if (scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
}

// ================================================
// ACTIVE NAVIGATION STATE
// ================================================
function updateActiveNav() {
    const scrollPosition = window.pageYOffset + 150;
    
    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        
        if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
}

// ================================================
// SCROLL REVEAL ANIMATIONS
// ================================================
function initScrollReveal() {
    const revealElements = document.querySelectorAll(
        '.timeline-item, .skill-category, .cert-card, .stat-card, .contact-card, .recognition-item, .blog-category-card'
    );
    
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                // Add staggered delay for grouped elements
                setTimeout(() => {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }, index * 50);
                
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    revealElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(40px)';
        el.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
        observer.observe(el);
    });
}

// ================================================
// HERO PARALLAX EFFECT
// ================================================
function updateHeroParallax() {
    const hero = document.querySelector('.hero');
    const scrollY = window.pageYOffset;
    
    if (hero && scrollY < 800) {
        const translateY = scrollY * 0.4;
        const opacity = 1 - (scrollY / 700);
        
        hero.style.transform = `translateY(${translateY}px)`;
        
        const heroContent = hero.querySelector('.hero-content');
        if (heroContent) {
            heroContent.style.opacity = Math.max(0, opacity);
        }
    }
}

// ================================================
// SCROLL TO TOP BUTTON
// ================================================
function initScrollToTop() {
    // Create button
    const scrollTopBtn = document.createElement('button');
    scrollTopBtn.id = 'scrollTopBtn';
    scrollTopBtn.innerHTML = '<i class="fas fa-arrow-up"></i>';
    scrollTopBtn.setAttribute('aria-label', 'Scroll to top');
    document.body.appendChild(scrollTopBtn);
    
    // Show/hide button
    function toggleScrollTopBtn() {
        if (window.pageYOffset > 400) {
            scrollTopBtn.classList.add('visible');
        } else {
            scrollTopBtn.classList.remove('visible');
        }
    }
    
    // Scroll to top on click
    scrollTopBtn.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
    
    // Return the toggle function for use in scroll handler
    return toggleScrollTopBtn;
}

// ================================================
// INTERACTIVE SKILL ITEMS
// ================================================
function initSkillInteractions() {
    document.querySelectorAll('.skill-item').forEach(item => {
        item.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-3px) scale(1.02)';
        });
        
        item.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
}

// ================================================
// CONTACT CARD INTERACTIONS
// ================================================
function initContactInteractions() {
    document.querySelectorAll('.contact-card').forEach(card => {
        const href = card.getAttribute('href');
        
        if (href && (href.startsWith('tel:') || href.startsWith('mailto:'))) {
            card.addEventListener('click', function() {
                showToast('Opening...');
            });
        }
    });
}

// ================================================
// TOAST NOTIFICATION
// ================================================
function showToast(message, duration = 2000) {
    const existing = document.querySelector('.toast-notification');
    if (existing) existing.remove();
    
    const toast = document.createElement('div');
    toast.className = 'toast-notification';
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 100px;
        left: 50%;
        transform: translateX(-50%) translateY(20px);
        background: rgba(15, 15, 15, 0.95);
        color: white;
        padding: 14px 28px;
        border-radius: 50px;
        font-size: 0.95rem;
        font-weight: 500;
        z-index: 9999;
        opacity: 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        font-family: 'Outfit', sans-serif;
    `;
    
    document.body.appendChild(toast);
    
    // Animate in
    requestAnimationFrame(() => {
        toast.style.opacity = '1';
        toast.style.transform = 'translateX(-50%) translateY(0)';
    });
    
    // Animate out
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(-50%) translateY(20px)';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ================================================
// KEYBOARD NAVIGATION
// ================================================
function initKeyboardNav() {
    document.addEventListener('keydown', (e) => {
        // ESC to close mobile menu
        if (e.key === 'Escape' && navMenu.classList.contains('active')) {
            navMenu.classList.remove('active');
            const icon = navToggle.querySelector('i');
            icon.classList.remove('fa-times');
            icon.classList.add('fa-bars');
        }
        
        // Ctrl+P to print
        if (e.key === 'p' && e.ctrlKey) {
            e.preventDefault();
            window.print();
        }
    });
}

// ================================================
// TYPING EFFECT FOR HERO (Optional)
// ================================================
function initTypingEffect() {
    const heroRole = document.querySelector('.hero-role');
    if (!heroRole) return;
    
    const roles = [
        'Senior Technical Consultant',
        'Cloud Architect',
        'DevSecOps Expert',
        'Trusted Advisor'
    ];
    
    let roleIndex = 0;
    let charIndex = 0;
    let isDeleting = false;
    let isPaused = false;
    
    function type() {
        const currentRole = roles[roleIndex];
        
        if (isPaused) {
            setTimeout(type, 2000);
            isPaused = false;
            isDeleting = true;
            return;
        }
        
        if (isDeleting) {
            heroRole.textContent = currentRole.substring(0, charIndex - 1);
            charIndex--;
            
            if (charIndex === 0) {
                isDeleting = false;
                roleIndex = (roleIndex + 1) % roles.length;
            }
        } else {
            heroRole.textContent = currentRole.substring(0, charIndex + 1);
            charIndex++;
            
            if (charIndex === currentRole.length) {
                isPaused = true;
            }
        }
        
        const typingSpeed = isDeleting ? 50 : 100;
        setTimeout(type, typingSpeed);
    }
    
    // Start typing after initial animation
    setTimeout(type, 3000);
}

// ================================================
// CONSOLIDATED SCROLL HANDLER
// ================================================
let scrollTicking = false;
let toggleScrollTopBtn = null;

function handleScroll() {
    updateNavbar();
    updateActiveNav();
    updateHeroParallax();
    if (toggleScrollTopBtn) toggleScrollTopBtn();
    scrollTicking = false;
}

function onScroll() {
    if (!scrollTicking) {
        window.requestAnimationFrame(handleScroll);
        scrollTicking = true;
    }
}

// ================================================
// INITIALIZE EVERYTHING
// ================================================
function init() {
    // Initialize all modules
    initMobileNav();
    initSmoothScroll();
    initScrollReveal();
    initSkillInteractions();
    initContactInteractions();
    initKeyboardNav();
    toggleScrollTopBtn = initScrollToTop();
    
    // Optional: Enable typing effect
    // initTypingEffect();
    
    // Add scroll listener
    window.addEventListener('scroll', onScroll, { passive: true });
    
    // Initial calls
    updateNavbar();
    updateActiveNav();
    
    // Log ready message
    console.log('🚀 Portfolio loaded successfully!');
    console.log('💼 Rakesh Kumar Mallam - Senior Technical Consultant');
}

// Run initialization
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
