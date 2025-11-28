# Rakesh Kumar Mallam - Portfolio Website

A modern, interactive portfolio website with dark mode, terminal interface, and dynamic content loading.

🌐 **Live Site:** [rmallam.github.io](https://rmallam.github.io)

---

## ✨ Features

- 🎨 **Modern UI** - Clean design with smooth animations
- 🌙 **Dark/Light Mode** - Toggle with one click
- 🖥️ **Interactive Terminal** - Type commands to explore
- 🎮 **CI/CD Pipeline Game** - Test your DevOps knowledge
- 📱 **Fully Responsive** - Works on all devices
- 📝 **Blog System** - Markdown-based blog posts
- 📄 **Resume Download** - One-click PDF download

---

## 🚀 How to Update the Website

### Quick Update (Recommended)

**Just edit `data.json` - no coding required!**

1. Open `data.json` in any editor
2. Make your changes
3. Push to GitHub

```bash
git add data.json
git commit -m "Update my info"
git push origin main
```

Or edit directly on GitHub:
1. Go to [data.json on GitHub](https://github.com/rmallam/rmallam.github.io/blob/main/data.json)
2. Click the ✏️ pencil icon
3. Edit and commit

---

## 📋 What to Edit in `data.json`

### Personal Information
```json
"personal": {
  "name": "Your Name",
  "role": "Your Job Title",
  "company": "Your Company",
  "tagline": "Your • Tagline • Here",
  "description": "Brief description about yourself",
  "availability": "Your availability status"
}
```

### Contact Details
```json
"contact": {
  "email": "your@email.com",
  "phone": "+1234567890",
  "linkedin": "https://linkedin.com/in/yourprofile",
  "github": "https://github.com/yourusername"
}
```

### About Section
```json
"about": {
  "intro": "First paragraph about you",
  "expertise": "Second paragraph about your expertise",
  "personal": "Third paragraph - personal touch",
  "stats": {
    "yearsExperience": "15+",
    "projectsDelivered": "50+",
    "certifications": "4"
  }
}
```

### Work Experience
```json
"experience": [
  {
    "title": "Job Title",
    "company": "Company Name",
    "icon": "fas fa-building",
    "period": "Jan 2020 - Present",
    "highlights": [
      "Achievement 1",
      "Achievement 2"
    ],
    "technologies": ["Tech1", "Tech2", "Tech3"]
  }
]
```

### Skills
```json
"skills": [
  {
    "category": "Category Name",
    "icon": "fas fa-cloud",
    "items": ["Skill1", "Skill2", "Skill3"]
  }
]
```

### Certifications
```json
"certifications": [
  {
    "name": "Certification Name",
    "issuer": "Issuing Organization",
    "icon": "fab fa-aws",
    "color": "aws"
  }
]
```
> Available colors: `redhat`, `aws`, `azure`

### Recognition/Awards
```json
"recognition": [
  {
    "title": "Award Title",
    "description": "Organization Name",
    "icon": "fas fa-trophy"
  }
]
```

---

## 📁 File Structure

```
rmallam.github.io/
├── index.html          # Main HTML (rarely needs editing)
├── styles.css          # All styling
├── data.json           # ⭐ EDIT THIS to update content
├── script.js           # Main JavaScript
├── dataLoader.js       # Loads data.json
├── terminal.js         # Terminal functionality
├── game.js             # CI/CD game
├── blogs.js            # Blog system
├── blogs/              # Blog posts (markdown)
│   ├── index.json      # Blog index
│   └── [categories]/   # Blog categories
└── resume/
    └── CV_MALLAM_RAKESHKUMAR.pdf  # Your resume
```

---

## 📝 Adding Blog Posts

1. Create a `.md` file in the appropriate `blogs/` subfolder
2. Add the filename to `blogs/index.json`
3. Push changes

Example blog structure:
```
blogs/
├── Machine Learning/
│   └── my_new_post_2025-01-15.md
└── index.json  ← Add filename here
```

---

## 📄 Updating Your Resume

Simply replace the PDF file:

```bash
cp /path/to/new/resume.pdf resume/CV_MALLAM_RAKESHKUMAR.pdf
git add resume/
git commit -m "Update resume"
git push origin main
```

---

## 🖥️ Terminal Commands

The interactive terminal supports these commands:

| Command | Description |
|---------|-------------|
| `help` | List all commands |
| `whoami` | About me |
| `skills` | Technical skills |
| `experience` | Work history |
| `certs` | Certifications |
| `achievements` | Awards & recognition |
| `contact` | Contact information |
| `hire` | Why hire me |
| `download` | Download resume |
| `social` | Social links |
| `joke` | Random dev joke |
| `clear` | Clear terminal |
| `exit` | Close terminal |

---

## 🧪 Testing Locally

```bash
# Start local server
cd rmallam.github.io
python3 -m http.server 8000

# Open in browser
open http://localhost:8000
```

---

## 🎨 Icons Reference

Find icons at [Font Awesome](https://fontawesome.com/icons)

Common icons:
- `fas fa-building` - Building
- `fas fa-university` - Bank/University
- `fab fa-redhat` - Red Hat
- `fab fa-aws` - AWS
- `fab fa-microsoft` - Microsoft
- `fas fa-trophy` - Trophy
- `fas fa-award` - Award
- `fas fa-star` - Star
- `fas fa-medal` - Medal

---

## 📞 Support

For any issues or questions, feel free to reach out!

---

Made with ❤️ by Rakesh Kumar Mallam

