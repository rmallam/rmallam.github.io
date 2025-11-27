// Blog Management System
const blogCategories = {};
let currentCategory = null;
let currentBlogPost = null;

// Initialize blogs on page load
document.addEventListener('DOMContentLoaded', () => {
    loadBlogStructure();
});

// Load blog structure from JSON index
async function loadBlogStructure() {
    try {
        const response = await fetch('blogs/index.json');
        const categories = await response.json();

        // Store categories
        Object.keys(categories).forEach(category => {
            if (categories[category].length > 0) {
                blogCategories[category] = categories[category];
            }
        });

        displayCategories();
    } catch (error) {
        console.error('Error loading blog structure:', error);
        document.getElementById('blog-categories').innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i>
                <p>Unable to load blog categories. Please try again later.</p>
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
        categoriesContainer.innerHTML = '<p class="no-blogs">No blog categories found</p>';
        return;
    }

    categoryNames.forEach(category => {
        const blogCount = blogCategories[category].length;
        if (blogCount === 0) return; // Skip empty categories

        const categoryCard = document.createElement('div');
        categoryCard.className = 'blog-category-card';
        categoryCard.innerHTML = `
            <div class="category-icon">
                <i class="fas fa-folder-open"></i>
            </div>
            <h3>${formatCategoryName(category)}</h3>
            <p>${blogCount} article${blogCount !== 1 ? 's' : ''}</p>
        `;
        categoryCard.onclick = () => showCategoryBlogs(category);
        categoriesContainer.appendChild(categoryCard);
    });
}

// Format category name for display
function formatCategoryName(name) {
    return name
        .split(/[_\s]+/)
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
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
    blogsContainer.innerHTML = '';

    for (const blogFile of blogs) {
        try {
            const response = await fetch(`blogs/${category}/${blogFile}`);
            const content = await response.text();
            
            const blogCard = document.createElement('div');
            blogCard.className = 'blog-card';
            
            const title = formatBlogTitle(blogFile);
            const summary = extractSummary(content);
            const date = extractDate(blogFile);

            blogCard.innerHTML = `
                <div class="blog-card-header">
                    <h4>${title}</h4>
                    ${date ? `<span class="blog-date"><i class="far fa-calendar"></i> ${date}</span>` : ''}
                </div>
                <p class="blog-summary">${summary}</p>
                <button class="read-more-btn" onclick="showBlogPost('${category}', '${blogFile}')">
                    Read More <i class="fas fa-arrow-right"></i>
                </button>
            `;
            blogsContainer.appendChild(blogCard);
        } catch (error) {
            console.error(`Error loading blog ${blogFile}:`, error);
        }
    }
}

// Format blog title from filename
function formatBlogTitle(filename) {
    return filename
        .replace('.md', '')
        .replace(/_/g, ' ')
        .replace(/(\d{4})-(\d{2})-(\d{2})/, '')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ')
        .trim();
}

// Extract date from filename
function extractDate(filename) {
    const dateMatch = filename.match(/(\d{4})-(\d{2})-(\d{2})/);
    if (dateMatch) {
        const [, year, month, day] = dateMatch;
        const date = new Date(year, month - 1, day);
        return date.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
    }
    return null;
}

// Extract summary from content
function extractSummary(content) {
    // Remove markdown headers
    const text = content.replace(/^#+\s+.+$/gm, '');
    // Get first paragraph
    const paragraphs = text.split('\n\n').filter(p => p.trim().length > 0);
    const summary = paragraphs[0] || '';
    // Limit to 150 characters
    return summary.length > 150 ? summary.substring(0, 150) + '...' : summary;
}

// Show individual blog post
async function showBlogPost(category, filename) {
    currentBlogPost = { category, filename };

    try {
        const response = await fetch(`blogs/${category}/${filename}`);
        const content = await response.text();

        document.getElementById('blog-list').style.display = 'none';
        document.getElementById('blog-post').style.display = 'block';

        const blogContent = document.getElementById('blog-content');
        const html = marked.parse(content);
        
        const title = formatBlogTitle(filename);
        const date = extractDate(filename);

        blogContent.innerHTML = `
            <div class="blog-post-header">
                <h2>${title}</h2>
                ${date ? `<p class="blog-post-date"><i class="far fa-calendar"></i> ${date}</p>` : ''}
            </div>
            <div class="blog-post-body">
                ${html}
            </div>
        `;

        // Scroll to top of blog post
        document.getElementById('blog-post').scrollIntoView({ behavior: 'smooth' });
    } catch (error) {
        console.error(`Error loading blog post:`, error);
        document.getElementById('blog-content').innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i>
                <p>Sorry, we couldn't load this blog post.</p>
            </div>
        `;
    }
}

// Navigate back to categories
function showCategories() {
    document.getElementById('blog-list').style.display = 'none';
    document.getElementById('blog-categories').style.display = 'grid';
    currentCategory = null;
}

// Navigate back to blog list
function backToList() {
    document.getElementById('blog-post').style.display = 'none';
    document.getElementById('blog-list').style.display = 'block';
    currentBlogPost = null;
}
