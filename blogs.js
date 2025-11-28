// ================================================
// BLOG MANAGEMENT SYSTEM
// ================================================

const blogCategories = {};
let currentCategory = null;
let currentBlogPost = null;

// Category icons mapping
const categoryIcons = {
    'AI_Generated': 'fa-robot',
    'Artificial Intelligence': 'fa-brain',
    'Machine Learning': 'fa-chart-line',
    'How-To': 'fa-book-open',
    'Openshift Virtualisation': 'fa-server',
    'skupper': 'fa-network-wired',
    'templates': 'fa-file-code',
    'default': 'fa-folder-open'
};

// Get icon for category
function getCategoryIcon(category) {
    return categoryIcons[category] || categoryIcons['default'];
}

// Initialize blogs on page load
document.addEventListener('DOMContentLoaded', () => {
    initBlogs();
});

// Initialize blogs
function initBlogs() {
    const categoriesContainer = document.getElementById('blog-categories');
    
    if (!categoriesContainer) {
        console.warn('Blog categories container not found');
        return;
    }
    
    // Show loading state
    categoriesContainer.innerHTML = `
        <div class="blog-loading" style="grid-column: 1 / -1; text-align: center; padding: 60px 20px;">
            <i class="fas fa-spinner fa-spin" style="font-size: 2.5rem; color: var(--primary-color); margin-bottom: 16px; display: block;"></i>
            <p style="color: var(--text-secondary); font-size: 1.1rem;">Loading blog categories...</p>
        </div>
    `;
    
    loadBlogStructure();
}

// Load blog structure from JSON index
async function loadBlogStructure() {
    const categoriesContainer = document.getElementById('blog-categories');
    
    try {
        const response = await fetch('blogs/index.json');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const categories = await response.json();

        // Store categories (filter out empty ones and templates)
        Object.keys(categories).forEach(category => {
            if (categories[category].length > 0 && category !== 'templates') {
                blogCategories[category] = categories[category];
            }
        });

        displayCategories();
        
    } catch (error) {
        console.error('Error loading blog structure:', error);
        
        // Check if it's a CORS/file protocol issue
        const isFileProtocol = window.location.protocol === 'file:';
        
        categoriesContainer.innerHTML = `
            <div class="error-message" style="grid-column: 1 / -1; text-align: center; padding: 60px 20px;">
                <i class="fas fa-exclamation-triangle" style="font-size: 3rem; color: var(--primary-color); margin-bottom: 16px; display: block;"></i>
                <h3 style="color: var(--text-primary); margin-bottom: 12px;">Unable to Load Blogs</h3>
                ${isFileProtocol ? `
                    <p style="color: var(--text-secondary); margin-bottom: 16px;">
                        You're opening this file directly. To view blogs, please use a local server:
                    </p>
                    <code style="background: var(--light-muted); padding: 12px 20px; border-radius: 8px; display: inline-block; font-family: 'JetBrains Mono', monospace; color: var(--primary-color);">
                        python3 -m http.server 8000
                    </code>
                    <p style="color: var(--text-muted); margin-top: 12px; font-size: 0.9rem;">
                        Then visit <a href="http://localhost:8000" style="color: var(--primary-color);">http://localhost:8000</a>
                    </p>
                ` : `
                    <p style="color: var(--text-secondary);">
                        There was an error loading the blog data. Please try again later.
                    </p>
                `}
            </div>
        `;
    }
}

// Display blog categories
function displayCategories() {
    const categoriesContainer = document.getElementById('blog-categories');
    categoriesContainer.innerHTML = '';

    const categoryNames = Object.keys(blogCategories);
    
    if (categoryNames.length === 0) {
        categoriesContainer.innerHTML = `
            <div class="no-blogs" style="grid-column: 1 / -1; text-align: center; padding: 60px 20px;">
                <i class="fas fa-folder-open" style="font-size: 3rem; color: var(--text-muted); margin-bottom: 16px; display: block;"></i>
                <p style="color: var(--text-secondary);">No blog categories found</p>
            </div>
        `;
        return;
    }

    // Sort categories - put AI_Generated first, then alphabetically
    categoryNames.sort((a, b) => {
        if (a === 'AI_Generated') return -1;
        if (b === 'AI_Generated') return 1;
        return a.localeCompare(b);
    });

    categoryNames.forEach((category, index) => {
        const blogCount = blogCategories[category].length;
        if (blogCount === 0) return;

        const categoryCard = document.createElement('div');
        categoryCard.className = 'blog-category-card';
        categoryCard.style.animationDelay = `${index * 0.1}s`;
        
        const icon = getCategoryIcon(category);
        
        categoryCard.innerHTML = `
            <div class="category-icon">
                <i class="fas ${icon}"></i>
            </div>
            <h3>${formatCategoryName(category)}</h3>
            <p>${blogCount} article${blogCount !== 1 ? 's' : ''}</p>
        `;
        
        categoryCard.addEventListener('click', () => showCategoryBlogs(category));
        categoriesContainer.appendChild(categoryCard);
    });
}

// Format category name for display
function formatCategoryName(name) {
    // Handle special cases
    if (name === 'AI_Generated') return 'AI Generated';
    if (name === 'skupper') return 'Skupper';
    
    return name
        .split(/[_\s]+/)
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
}

// Show blogs in a category
async function showCategoryBlogs(category) {
    currentCategory = category;
    const blogs = blogCategories[category];

    document.getElementById('blog-categories').style.display = 'none';
    document.getElementById('blog-list').style.display = 'block';
    document.getElementById('category-title').textContent = formatCategoryName(category);

    const blogsContainer = document.getElementById('blogs-container');
    
    // Show loading state
    blogsContainer.innerHTML = `
        <div class="blog-loading" style="grid-column: 1 / -1; text-align: center; padding: 40px 20px;">
            <i class="fas fa-spinner fa-spin" style="font-size: 2rem; color: var(--primary-color);"></i>
            <p style="color: var(--text-secondary); margin-top: 12px;">Loading articles...</p>
        </div>
    `;

    // Sort blogs by date (newest first)
    const sortedBlogs = [...blogs].sort((a, b) => {
        const dateA = extractDateForSort(a);
        const dateB = extractDateForSort(b);
        return dateB - dateA;
    });

    // Clear loading and load blogs
    blogsContainer.innerHTML = '';
    
    let loadedCount = 0;
    const maxToShow = 20; // Limit for performance

    for (const blogFile of sortedBlogs.slice(0, maxToShow)) {
        try {
            const response = await fetch(`blogs/${category}/${blogFile}`);
            
            if (!response.ok) {
                console.warn(`Failed to load ${blogFile}`);
                continue;
            }
            
            const content = await response.text();
            
            const blogCard = document.createElement('div');
            blogCard.className = 'blog-card';
            blogCard.style.animationDelay = `${loadedCount * 0.05}s`;
            
            const title = formatBlogTitle(blogFile);
            const summary = extractSummary(content);
            const date = extractDate(blogFile);

            blogCard.innerHTML = `
                <div class="blog-card-header">
                    <h4>${title}</h4>
                    ${date ? `<span class="blog-date"><i class="far fa-calendar-alt"></i> ${date}</span>` : ''}
                </div>
                <p class="blog-summary">${escapeHtml(summary)}</p>
                <button class="read-more-btn" onclick="showBlogPost('${escapeAttr(category)}', '${escapeAttr(blogFile)}')">
                    Read More <i class="fas fa-arrow-right"></i>
                </button>
            `;
            blogsContainer.appendChild(blogCard);
            loadedCount++;
            
        } catch (error) {
            console.error(`Error loading blog ${blogFile}:`, error);
        }
    }

    // Show message if more articles exist
    if (sortedBlogs.length > maxToShow) {
        const moreInfo = document.createElement('div');
        moreInfo.style.cssText = 'grid-column: 1 / -1; text-align: center; padding: 20px; color: var(--text-muted);';
        moreInfo.innerHTML = `<p>Showing ${maxToShow} of ${sortedBlogs.length} articles</p>`;
        blogsContainer.appendChild(moreInfo);
    }

    // If no blogs loaded, show message
    if (loadedCount === 0) {
        blogsContainer.innerHTML = `
            <div class="no-blogs" style="grid-column: 1 / -1; text-align: center; padding: 40px 20px;">
                <i class="fas fa-file-alt" style="font-size: 2.5rem; color: var(--text-muted); margin-bottom: 12px; display: block;"></i>
                <p style="color: var(--text-secondary);">No articles found in this category</p>
            </div>
        `;
    }

    // Scroll to blog list
    document.getElementById('blog-list').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Format blog title from filename
function formatBlogTitle(filename) {
    return filename
        .replace('.md', '')
        .replace(/_/g, ' ')
        .replace(/\s*\d{4}-\d{2}-\d{2}\s*/, ' ')
        .split(' ')
        .filter(word => word.length > 0)
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ')
        .trim();
}

// Extract date from filename for sorting
function extractDateForSort(filename) {
    const dateMatch = filename.match(/(\d{4})-(\d{2})-(\d{2})/);
    if (dateMatch) {
        return new Date(dateMatch[1], dateMatch[2] - 1, dateMatch[3]).getTime();
    }
    return 0;
}

// Extract date from filename for display
function extractDate(filename) {
    const dateMatch = filename.match(/(\d{4})-(\d{2})-(\d{2})/);
    if (dateMatch) {
        const [, year, month, day] = dateMatch;
        const date = new Date(year, month - 1, day);
        return date.toLocaleDateString('en-US', { 
            year: 'numeric', 
            month: 'short', 
            day: 'numeric' 
        });
    }
    return null;
}

// Extract summary from content
function extractSummary(content) {
    // Remove YAML frontmatter if present
    let text = content.replace(/^---[\s\S]*?---\n?/, '');
    
    // Remove markdown headers
    text = text.replace(/^#+\s+.+$/gm, '');
    
    // Remove markdown formatting
    text = text.replace(/\*\*|__/g, '');
    text = text.replace(/\*|_/g, '');
    text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
    text = text.replace(/`[^`]+`/g, '');
    text = text.replace(/```[\s\S]*?```/g, '');
    
    // Get first meaningful paragraph
    const paragraphs = text.split(/\n\n+/).filter(p => {
        const trimmed = p.trim();
        return trimmed.length > 20 && !trimmed.startsWith('-') && !trimmed.startsWith('|');
    });
    
    const summary = paragraphs[0] || '';
    const cleanSummary = summary.replace(/\n/g, ' ').trim();
    
    // Limit to 180 characters
    return cleanSummary.length > 180 
        ? cleanSummary.substring(0, 180).replace(/\s+\S*$/, '') + '...' 
        : cleanSummary;
}

// Escape HTML entities
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Escape attribute value
function escapeAttr(text) {
    return text.replace(/'/g, "\\'").replace(/"/g, '\\"');
}

// Show individual blog post
async function showBlogPost(category, filename) {
    currentBlogPost = { category, filename };

    const blogContent = document.getElementById('blog-content');
    
    // Show loading state
    blogContent.innerHTML = `
        <div style="text-align: center; padding: 60px 20px;">
            <i class="fas fa-spinner fa-spin" style="font-size: 2.5rem; color: var(--primary-color);"></i>
            <p style="color: var(--text-secondary); margin-top: 16px;">Loading article...</p>
        </div>
    `;

    document.getElementById('blog-list').style.display = 'none';
    document.getElementById('blog-post').style.display = 'block';

    try {
        const response = await fetch(`blogs/${category}/${filename}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        let content = await response.text();
        
        // Remove YAML frontmatter if present
        content = content.replace(/^---[\s\S]*?---\n?/, '');
        
        // Parse markdown
        const html = marked.parse(content);
        
        const title = formatBlogTitle(filename);
        const date = extractDate(filename);

        blogContent.innerHTML = `
            <div class="blog-post-header">
                <h2>${escapeHtml(title)}</h2>
                ${date ? `<p class="blog-post-date"><i class="far fa-calendar-alt"></i> ${date}</p>` : ''}
            </div>
            <div class="blog-post-body">
                ${html}
            </div>
        `;

        // Scroll to top of blog post
        document.getElementById('blog-post').scrollIntoView({ behavior: 'smooth', block: 'start' });
        
    } catch (error) {
        console.error(`Error loading blog post:`, error);
        blogContent.innerHTML = `
            <div class="error-message" style="text-align: center; padding: 60px 20px;">
                <i class="fas fa-exclamation-triangle" style="font-size: 3rem; color: var(--primary-color); margin-bottom: 16px; display: block;"></i>
                <h3 style="color: var(--text-primary); margin-bottom: 12px;">Unable to Load Article</h3>
                <p style="color: var(--text-secondary);">Sorry, we couldn't load this blog post. Please try again.</p>
                <button onclick="backToList()" class="read-more-btn" style="margin-top: 20px;">
                    <i class="fas fa-arrow-left"></i> Go Back
                </button>
            </div>
        `;
    }
}

// Navigate back to categories
function showCategories() {
    document.getElementById('blog-list').style.display = 'none';
    document.getElementById('blog-post').style.display = 'none';
    document.getElementById('blog-categories').style.display = 'grid';
    currentCategory = null;
    currentBlogPost = null;
    
    // Scroll to blogs section
    document.getElementById('blogs').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Navigate back to blog list
function backToList() {
    document.getElementById('blog-post').style.display = 'none';
    document.getElementById('blog-list').style.display = 'block';
    currentBlogPost = null;
    
    // Scroll to blog list
    document.getElementById('blog-list').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Make functions globally available
window.showCategories = showCategories;
window.backToList = backToList;
window.showBlogPost = showBlogPost;

console.log('📚 Blog system loaded!');
