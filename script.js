// Smooth scrolling for navigation links with proper offset
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            const navbarHeight = 70;
            const additionalPadding = 20;
            const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - navbarHeight - additionalPadding;
            
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });
        }
    });
});

// Navbar and scroll effects (handled in consolidated scroll handler below)

// Intersection Observer for fade-in animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all sections and cards
document.querySelectorAll('.timeline-item, .skill-category, .cert-card, .stat-card').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(30px)';
    el.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
    observer.observe(el);
});

// Active navigation state function
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-menu a');

function updateActiveNav() {
    let current = '';
    const scrollPosition = window.pageYOffset + 150; // Account for navbar height
    
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

// Add typing effect to hero title (optional enhancement)
function typeWriter(element, text, speed = 100) {
    let i = 0;
    element.textContent = '';
    
    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }
    
    type();
}

// Parallax effect (handled in consolidated scroll handler)

// Skills hover effect - show proficiency level
document.querySelectorAll('.skill-item').forEach(item => {
    item.addEventListener('mouseenter', function() {
        this.style.transform = 'scale(1.05)';
    });
    
    item.addEventListener('mouseleave', function() {
        this.style.transform = 'scale(1)';
    });
});

// Add click to copy functionality for contact info
document.querySelectorAll('.contact-card').forEach(card => {
    const href = card.getAttribute('href');
    
    if (href && (href.startsWith('tel:') || href.startsWith('mailto:'))) {
        card.addEventListener('click', function(e) {
            const value = href.startsWith('tel:') 
                ? href.replace('tel:', '') 
                : href.replace('mailto:', '');
            
            // Show a tooltip
            const tooltip = document.createElement('div');
            tooltip.textContent = 'Clicked! Opening...';
            tooltip.style.cssText = `
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 1rem 2rem;
                border-radius: 5px;
                z-index: 9999;
                animation: fadeOut 1s ease-out forwards;
            `;
            document.body.appendChild(tooltip);
            
            setTimeout(() => {
                tooltip.remove();
            }, 1000);
        });
    }
});

// Add CSS for fade out animation
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }
`;
document.head.appendChild(style);

// Print-friendly version
function preparePrint() {
    window.print();
}

// Add keyboard navigation
document.addEventListener('keydown', (e) => {
    // Press P to print
    if (e.key === 'p' && e.ctrlKey) {
        e.preventDefault();
        preparePrint();
    }
});

// Enhanced scroll to top button
const scrollTopBtn = document.createElement('button');
scrollTopBtn.id = 'scrollTopBtn';
scrollTopBtn.innerHTML = '<i class="fas fa-arrow-up"></i>';
scrollTopBtn.setAttribute('aria-label', 'Scroll to top');

scrollTopBtn.addEventListener('click', () => {
    window.scrollTo({ 
        top: 0, 
        behavior: 'smooth' 
    });
});

document.body.appendChild(scrollTopBtn);

// Show/hide scroll to top button function
function toggleScrollTopBtn() {
    if (window.pageYOffset > 300) {
        scrollTopBtn.classList.add('visible');
    } else {
        scrollTopBtn.classList.remove('visible');
    }
}

// Mobile menu toggle (if you add hamburger menu later)
function initMobileMenu() {
    const nav = document.querySelector('.nav-menu');
    const hamburger = document.createElement('div');
    hamburger.className = 'hamburger';
    hamburger.innerHTML = '<i class="fas fa-bars"></i>';
    
    if (window.innerWidth <= 768) {
        document.querySelector('.navbar .container').appendChild(hamburger);
        
        hamburger.addEventListener('click', () => {
            nav.classList.toggle('active');
        });
    }
}

// Initialize on load
window.addEventListener('load', () => {
    initMobileMenu();
    
    // Add fade-in effect to hero
    document.querySelector('.hero').style.opacity = '1';
});

// Consolidated scroll handler for better performance
let scrollTicking = false;

function handleScroll() {
    const currentScroll = window.pageYOffset;
    
    // Navbar effects
    const navbar = document.querySelector('.navbar');
    if (currentScroll > 100) {
        navbar.style.background = 'rgba(26, 26, 26, 0.98)';
        navbar.style.backdropFilter = 'blur(10px)';
        navbar.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.3)';
    } else {
        navbar.style.background = 'var(--dark-bg)';
        navbar.style.backdropFilter = 'none';
        navbar.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';
    }
    
    // Update active navigation
    updateActiveNav();
    
    // Toggle scroll to top button
    toggleScrollTopBtn();
    
    // Parallax effect
    const hero = document.querySelector('.hero');
    if (hero && currentScroll < 800) {
        hero.style.transform = `translateY(${currentScroll * 0.3}px)`;
        hero.style.opacity = 1 - (currentScroll / 800);
    }
    
    scrollTicking = false;
}

window.addEventListener('scroll', () => {
    if (!scrollTicking) {
        window.requestAnimationFrame(handleScroll);
        scrollTicking = true;
    }
});

console.log('🚀 Portfolio loaded successfully!');
console.log('💼 Rakesh Kumar Mallam - Senior Technical Consultant');
