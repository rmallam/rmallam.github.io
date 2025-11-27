# How to Contribute a Blog

This guide provides simple steps to contribute a blog post to our website.

## Submission Process

### 1. Fork the Repository

Start by forking the repository:
```bash
https://github.com/rmallam/blogs
```

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR-USERNAME/blogs.git
cd blogs
```

### 3. Create Your Blog Post

Write your blog post in Markdown format. Some guidelines:

- Use clear, concise language
- Include relevant images in the **public** folder
- Add code snippets when necessary
- Structure your content with headings
- Include a brief author bio at the end (optional)

### 4. Add Your Blog to the Repository

Place your Markdown file in the `src/blogs` folder:

```bash
# Create a new branch for your blog
git checkout -b blog/your-blog-title

# Add your markdown file to the blogs folder
# Example:
# my-awesome-blog.md
```

### 5. Submit a Pull Request

Once your blog is ready:

```bash
# Commit your changes
git add .
git commit -m "Add new blog: Your Blog Title"

# Push to your fork
git push origin blog/your-blog-title
```

Then go to the original repository on GitHub and create a pull request from your branch.

### 6. Review Process

After submitting your PR:
- The maintainers will review your submission
- They may suggest edits or improvements
- Once approved, your blog will be published automatically

## That's It!

The rest of the process will be handled by the site maintainers. Your blog will be integrated into the website and published once approved.

Thank you for contributing!
