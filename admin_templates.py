#!/usr/bin/env python3
"""
Admin Templates for Golden Coyotes Platform
HTML templates for admin panel functionality
"""

ADMIN_LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Login - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .admin-card { background: rgba(255,255,255,0.95); backdrop-filter: blur(10px); }
        .btn-admin { background: #dc3545; border-color: #dc3545; }
        .btn-admin:hover { background: #c82333; border-color: #bd2130; }
    </style>
</head>
<body class="d-flex align-items-center">
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-6 col-lg-4">
                <div class="card admin-card shadow-lg border-0">
                    <div class="card-header bg-danger text-white text-center">
                        <h3><i class="fas fa-shield-alt"></i> Admin Access</h3>
                        <p class="mb-0">Golden Coyotes Platform</p>
                    </div>
                    <div class="card-body p-5">
                        <form id="adminLoginForm">
                            <div class="mb-3">
                                <label for="email" class="form-label"><i class="fas fa-envelope"></i> Admin Email</label>
                                <input type="email" class="form-control" id="email" name="email" required>
                            </div>
                            <div class="mb-4">
                                <label for="password" class="form-label"><i class="fas fa-lock"></i> Password</label>
                                <input type="password" class="form-control" id="password" name="password" required>
                            </div>
                            <button type="submit" class="btn btn-admin w-100">
                                <i class="fas fa-sign-in-alt"></i> Admin Login
                            </button>
                        </form>
                        <hr>
                        <div class="text-center">
                            <small class="text-muted">
                                <i class="fas fa-info-circle"></i> Admin credentials required
                            </small>
                        </div>
                    </div>
                </div>
                
                <div class="text-center mt-3">
                    <a href="/" class="text-white">
                        <i class="fas fa-arrow-left"></i> Back to Public Site
                    </a>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.getElementById('adminLoginForm').onsubmit = async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData);
            
            try {
                const response = await fetch('/admin/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    window.location.href = result.redirect;
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Network error occurred');
            }
        };
    </script>
</body>
</html>
'''

ADMIN_DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Dashboard - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .admin-sidebar { min-height: 100vh; background: linear-gradient(180deg, #dc3545 0%, #c82333 100%); }
        .content { background-color: #f8f9fa; min-height: 100vh; }
        .stat-card { border-left: 4px solid #dc3545; }
        .activity-item { border-left: 3px solid #007bff; padding-left: 15px; margin-bottom: 15px; }
        .table-responsive { max-height: 400px; overflow-y: auto; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Admin Sidebar -->
            <div class="col-md-2 admin-sidebar text-white p-4">
                <h3 class="mb-4"><i class="fas fa-shield-alt"></i> Admin Panel</h3>
                <nav class="nav flex-column">
                    <a class="nav-link text-white active" href="/admin/dashboard">
                        <i class="fas fa-tachometer-alt"></i> Dashboard
                    </a>
                    <a class="nav-link text-white" href="/admin/users">
                        <i class="fas fa-users"></i> Users
                    </a>
                    <a class="nav-link text-white" href="/admin/opportunities">
                        <i class="fas fa-briefcase"></i> Opportunities
                    </a>
                    <a class="nav-link text-white" href="/admin/logs">
                        <i class="fas fa-clipboard-list"></i> Activity Logs
                    </a>
                    <hr>
                    <a class="nav-link text-white" href="/">
                        <i class="fas fa-globe"></i> Public Site
                    </a>
                    <a class="nav-link text-white" href="/logout">
                        <i class="fas fa-sign-out-alt"></i> Logout
                    </a>
                </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 content p-4">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h1><i class="fas fa-chart-line"></i> Platform Statistics</h1>
                    <div class="badge bg-danger fs-6">
                        <i class="fas fa-shield-alt"></i> Admin Level {{ session.admin_level }}
                    </div>
                </div>
                
                <!-- Statistics Cards -->
                <div class="row mb-5">
                    <div class="col-md-3">
                        <div class="card stat-card border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-users fa-2x text-primary mb-2"></i>
                                <h3 class="text-primary">{{ stats.total_users or 0 }}</h3>
                                <p class="mb-0">Total Users</p>
                                <small class="text-muted">+{{ stats.new_users_30d or 0 }} this month</small>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card stat-card border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-briefcase fa-2x text-success mb-2"></i>
                                <h3 class="text-success">{{ stats.total_opportunities or 0 }}</h3>
                                <p class="mb-0">Opportunities</p>
                                <small class="text-muted">+{{ stats.new_opportunities_30d or 0 }} this month</small>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card stat-card border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-handshake fa-2x text-warning mb-2"></i>
                                <h3 class="text-warning">{{ stats.total_connections or 0 }}</h3>
                                <p class="mb-0">Connections</p>
                                <small class="text-muted">+{{ stats.new_connections_30d or 0 }} this month</small>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card stat-card border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-chart-line fa-2x text-info mb-2"></i>
                                <h3 class="text-info">{{ stats.active_users_7d or 0 }}</h3>
                                <p class="mb-0">Active Users</p>
                                <small class="text-muted">Last 7 days</small>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <!-- Industry Breakdown -->
                    <div class="col-md-6">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-primary text-white">
                                <h5><i class="fas fa-industry"></i> Top Industries</h5>
                            </div>
                            <div class="card-body">
                                {% if stats.top_industries %}
                                    {% for industry, count in stats.top_industries %}
                                    <div class="d-flex justify-content-between align-items-center mb-2">
                                        <span>{{ industry or 'Not Specified' }}</span>
                                        <span class="badge bg-primary">{{ count }}</span>
                                    </div>
                                    {% endfor %}
                                {% else %}
                                    <p class="text-muted">No industry data available</p>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                    
                    <!-- Location Breakdown -->
                    <div class="col-md-6">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-success text-white">
                                <h5><i class="fas fa-map-marker-alt"></i> Top Locations</h5>
                            </div>
                            <div class="card-body">
                                {% if stats.top_locations %}
                                    {% for location, count in stats.top_locations %}
                                    <div class="d-flex justify-content-between align-items-center mb-2">
                                        <span>{{ location or 'Not Specified' }}</span>
                                        <span class="badge bg-success">{{ count }}</span>
                                    </div>
                                    {% endfor %}
                                {% else %}
                                    <p class="text-muted">No location data available</p>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Recent Activity -->
                <div class="row mt-4">
                    <div class="col-md-6">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-info text-white">
                                <h5><i class="fas fa-user-plus"></i> Recent Users</h5>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    <table class="table table-sm">
                                        <thead>
                                            <tr>
                                                <th>Name</th>
                                                <th>Email</th>
                                                <th>Industry</th>
                                                <th>Status</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {% for user in recent_users %}
                                            <tr>
                                                <td>{{ user.name }}</td>
                                                <td>{{ user.email }}</td>
                                                <td>{{ user.industry or 'N/A' }}</td>
                                                <td>
                                                    <span class="badge {% if user.status == 'active' %}bg-success{% else %}bg-warning{% endif %}">
                                                        {{ user.status }}
                                                    </span>
                                                </td>
                                            </tr>
                                            {% endfor %}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-warning text-dark">
                                <h5><i class="fas fa-history"></i> Recent Admin Activity</h5>
                            </div>
                            <div class="card-body">
                                <div style="max-height: 300px; overflow-y: auto;">
                                    {% for log in recent_logs %}
                                    <div class="activity-item">
                                        <small class="text-muted">{{ log.created_at }}</small>
                                        <br>
                                        <strong>{{ log.admin_name }}</strong>: {{ log.action }}
                                        {% if log.details %}
                                        <br><small class="text-muted">{{ log.details }}</small>
                                        {% endif %}
                                    </div>
                                    {% endfor %}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

ADMIN_USERS_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>User Management - Golden Coyotes Admin</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .admin-sidebar { min-height: 100vh; background: linear-gradient(180deg, #dc3545 0%, #c82333 100%); }
        .content { background-color: #f8f9fa; min-height: 100vh; }
        .user-row:hover { background-color: #f1f3f4; }
        .status-active { color: #28a745; }
        .status-inactive { color: #ffc107; }
        .status-suspended { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Admin Sidebar -->
            <div class="col-md-2 admin-sidebar text-white p-4">
                <h3 class="mb-4"><i class="fas fa-shield-alt"></i> Admin Panel</h3>
                <nav class="nav flex-column">
                    <a class="nav-link text-white" href="/admin/dashboard">
                        <i class="fas fa-tachometer-alt"></i> Dashboard
                    </a>
                    <a class="nav-link text-white active" href="/admin/users">
                        <i class="fas fa-users"></i> Users
                    </a>
                    <a class="nav-link text-white" href="/admin/opportunities">
                        <i class="fas fa-briefcase"></i> Opportunities
                    </a>
                    <a class="nav-link text-white" href="/admin/logs">
                        <i class="fas fa-clipboard-list"></i> Activity Logs
                    </a>
                    <hr>
                    <a class="nav-link text-white" href="/">
                        <i class="fas fa-globe"></i> Public Site
                    </a>
                    <a class="nav-link text-white" href="/logout">
                        <i class="fas fa-sign-out-alt"></i> Logout
                    </a>
                </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 content p-4">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h1><i class="fas fa-users"></i> User Management</h1>
                    <div class="d-flex gap-2">
                        <input type="text" class="form-control" id="searchUsers" placeholder="Search users...">
                        <button class="btn btn-outline-primary">
                            <i class="fas fa-search"></i>
                        </button>
                    </div>
                </div>
                
                <!-- Users Table -->
                <div class="card border-0 shadow-sm">
                    <div class="card-header bg-primary text-white">
                        <h5><i class="fas fa-table"></i> All Users ({{ users|length }})</h5>
                    </div>
                    <div class="card-body p-0">
                        <div class="table-responsive" style="max-height: 600px;">
                            <table class="table table-hover mb-0">
                                <thead class="bg-light sticky-top">
                                    <tr>
                                        <th>Name</th>
                                        <th>Email</th>
                                        <th>Industry</th>
                                        <th>Location</th>
                                        <th>Role</th>
                                        <th>Status</th>
                                        <th>Joined</th>
                                        <th>Last Login</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for user in users %}
                                    <tr class="user-row" data-user-id="{{ user.id }}">
                                        <td>
                                            <strong>{{ user.name }}</strong>
                                            {% if user.user_role == 'admin' %}
                                            <span class="badge bg-danger ms-1">Admin</span>
                                            {% endif %}
                                        </td>
                                        <td>{{ user.email }}</td>
                                        <td>{{ user.industry or 'N/A' }}</td>
                                        <td>{{ user.location or 'N/A' }}</td>
                                        <td>
                                            <span class="badge {% if user.user_role == 'admin' %}bg-danger{% else %}bg-secondary{% endif %}">
                                                {{ user.user_role or 'user' }}
                                            </span>
                                        </td>
                                        <td>
                                            <span class="status-{{ user.status }}">
                                                <i class="fas fa-circle"></i> {{ user.status }}
                                            </span>
                                        </td>
                                        <td>{{ user.created_at[:10] if user.created_at else 'N/A' }}</td>
                                        <td>{{ user.last_login[:10] if user.last_login else 'Never' }}</td>
                                        <td>
                                            <div class="dropdown">
                                                <button class="btn btn-sm btn-outline-secondary dropdown-toggle" 
                                                        data-bs-toggle="dropdown">
                                                    Actions
                                                </button>
                                                <ul class="dropdown-menu">
                                                    {% if user.status == 'active' %}
                                                    <li><a class="dropdown-item text-warning" href="#" 
                                                           onclick="updateUserStatus('{{ user.id }}', 'suspended')">
                                                        <i class="fas fa-ban"></i> Suspend User
                                                    </a></li>
                                                    {% else %}
                                                    <li><a class="dropdown-item text-success" href="#" 
                                                           onclick="updateUserStatus('{{ user.id }}', 'active')">
                                                        <i class="fas fa-check"></i> Activate User
                                                    </a></li>
                                                    {% endif %}
                                                    <li><a class="dropdown-item" href="#" onclick="viewUserDetails('{{ user.id }}')">
                                                        <i class="fas fa-eye"></i> View Details
                                                    </a></li>
                                                </ul>
                                            </div>
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        async function updateUserStatus(userId, newStatus) {
            if (!confirm(`Are you sure you want to change this user's status to ${newStatus}?`)) {
                return;
            }
            
            try {
                const response = await fetch('/admin/api/user-status', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: userId, status: newStatus })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    location.reload();
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Network error occurred');
            }
        }
        
        function viewUserDetails(userId) {
            // Future implementation for detailed user view
            alert('User details view - Coming soon!');
        }
        
        // Search functionality
        document.getElementById('searchUsers').addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            const rows = document.querySelectorAll('.user-row');
            
            rows.forEach(row => {
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(searchTerm) ? '' : 'none';
            });
        });
    </script>
</body>
</html>
'''

ADMIN_OPPORTUNITIES_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Opportunity Management - Golden Coyotes Admin</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .admin-sidebar { min-height: 100vh; background: linear-gradient(180deg, #dc3545 0%, #c82333 100%); }
        .content { background-color: #f8f9fa; min-height: 100vh; }
        .opportunity-card { transition: transform 0.2s; }
        .opportunity-card:hover { transform: translateY(-2px); }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Admin Sidebar -->
            <div class="col-md-2 admin-sidebar text-white p-4">
                <h3 class="mb-4"><i class="fas fa-shield-alt"></i> Admin Panel</h3>
                <nav class="nav flex-column">
                    <a class="nav-link text-white" href="/admin/dashboard">
                        <i class="fas fa-tachometer-alt"></i> Dashboard
                    </a>
                    <a class="nav-link text-white" href="/admin/users">
                        <i class="fas fa-users"></i> Users
                    </a>
                    <a class="nav-link text-white active" href="/admin/opportunities">
                        <i class="fas fa-briefcase"></i> Opportunities
                    </a>
                    <a class="nav-link text-white" href="/admin/logs">
                        <i class="fas fa-clipboard-list"></i> Activity Logs
                    </a>
                    <hr>
                    <a class="nav-link text-white" href="/">
                        <i class="fas fa-globe"></i> Public Site
                    </a>
                    <a class="nav-link text-white" href="/logout">
                        <i class="fas fa-sign-out-alt"></i> Logout
                    </a>
                </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 content p-4">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h1><i class="fas fa-briefcase"></i> Opportunity Management</h1>
                    <div class="d-flex gap-2">
                        <select class="form-select" id="filterType">
                            <option value="">All Types</option>
                            <option value="partnership">Partnership</option>
                            <option value="investment">Investment</option>
                            <option value="buyer">Buyer</option>
                            <option value="seller">Seller</option>
                            <option value="collaboration">Collaboration</option>
                        </select>
                        <input type="text" class="form-control" id="searchOpportunities" placeholder="Search opportunities...">
                    </div>
                </div>
                
                <!-- Opportunities Grid -->
                <div class="row">
                    {% for opportunity in opportunities %}
                    <div class="col-md-6 col-lg-4 mb-4 opportunity-item" 
                         data-type="{{ opportunity.type }}" 
                         data-search="{{ (opportunity.title + ' ' + opportunity.description + ' ' + opportunity.creator_name).lower() }}">
                        <div class="card opportunity-card border-0 shadow-sm h-100">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <span class="badge {% if opportunity.type == 'partnership' %}bg-primary{% elif opportunity.type == 'investment' %}bg-success{% elif opportunity.type == 'buyer' %}bg-info{% elif opportunity.type == 'seller' %}bg-warning{% else %}bg-secondary{% endif %}">
                                    {{ opportunity.type|title }}
                                </span>
                                <div class="dropdown">
                                    <button class="btn btn-sm btn-outline-secondary dropdown-toggle" 
                                            data-bs-toggle="dropdown">
                                        <i class="fas fa-cog"></i>
                                    </button>
                                    <ul class="dropdown-menu">
                                        <li><a class="dropdown-item text-warning" href="#" 
                                               onclick="moderateOpportunity('{{ opportunity.id }}', 'flag')">
                                            <i class="fas fa-flag"></i> Flag for Review
                                        </a></li>
                                        <li><a class="dropdown-item text-danger" href="#" 
                                               onclick="moderateOpportunity('{{ opportunity.id }}', 'disable')">
                                            <i class="fas fa-ban"></i> Disable
                                        </a></li>
                                    </ul>
                                </div>
                            </div>
                            <div class="card-body">
                                <h6 class="card-title">{{ opportunity.title }}</h6>
                                <p class="card-text text-muted small">
                                    {{ opportunity.description[:100] }}{% if opportunity.description|length > 100 %}...{% endif %}
                                </p>
                                <div class="row text-center">
                                    <div class="col-6">
                                        <small class="text-muted">Creator</small>
                                        <br><strong>{{ opportunity.creator_name }}</strong>
                                    </div>
                                    <div class="col-6">
                                        <small class="text-muted">Industry</small>
                                        <br><strong>{{ opportunity.industry or 'N/A' }}</strong>
                                    </div>
                                </div>
                            </div>
                            <div class="card-footer bg-light">
                                <div class="d-flex justify-content-between align-items-center">
                                    <small class="text-muted">
                                        <i class="fas fa-calendar"></i> 
                                        {{ opportunity.created_at[:10] if opportunity.created_at else 'N/A' }}
                                    </small>
                                    <small class="text-muted">
                                        <i class="fas fa-eye"></i> 
                                        {{ opportunity.views or 0 }} views
                                    </small>
                                </div>
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
                
                {% if not opportunities %}
                <div class="text-center mt-5">
                    <i class="fas fa-briefcase fa-3x text-muted mb-3"></i>
                    <h5 class="text-muted">No opportunities found</h5>
                    <p class="text-muted">Opportunities will appear here as users create them.</p>
                </div>
                {% endif %}
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function moderateOpportunity(oppId, action) {
            if (!confirm(`Are you sure you want to ${action} this opportunity?`)) {
                return;
            }
            
            // Future implementation for opportunity moderation
            alert(`Opportunity ${action} - Coming soon!`);
        }
        
        // Filter functionality
        document.getElementById('filterType').addEventListener('change', function(e) {
            const filterType = e.target.value;
            const items = document.querySelectorAll('.opportunity-item');
            
            items.forEach(item => {
                const itemType = item.dataset.type;
                item.style.display = (!filterType || itemType === filterType) ? 'block' : 'none';
            });
        });
        
        // Search functionality
        document.getElementById('searchOpportunities').addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            const items = document.querySelectorAll('.opportunity-item');
            
            items.forEach(item => {
                const searchData = item.dataset.search;
                item.style.display = searchData.includes(searchTerm) ? 'block' : 'none';
            });
        });
    </script>
</body>
</html>
'''

ADMIN_LOGS_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Activity Logs - Golden Coyotes Admin</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .admin-sidebar { min-height: 100vh; background: linear-gradient(180deg, #dc3545 0%, #c82333 100%); }
        .content { background-color: #f8f9fa; min-height: 100vh; }
        .log-entry { border-left: 4px solid #007bff; background: white; }
        .log-critical { border-left-color: #dc3545; }
        .log-warning { border-left-color: #ffc107; }
        .log-success { border-left-color: #28a745; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Admin Sidebar -->
            <div class="col-md-2 admin-sidebar text-white p-4">
                <h3 class="mb-4"><i class="fas fa-shield-alt"></i> Admin Panel</h3>
                <nav class="nav flex-column">
                    <a class="nav-link text-white" href="/admin/dashboard">
                        <i class="fas fa-tachometer-alt"></i> Dashboard
                    </a>
                    <a class="nav-link text-white" href="/admin/users">
                        <i class="fas fa-users"></i> Users
                    </a>
                    <a class="nav-link text-white" href="/admin/opportunities">
                        <i class="fas fa-briefcase"></i> Opportunities
                    </a>
                    <a class="nav-link text-white active" href="/admin/logs">
                        <i class="fas fa-clipboard-list"></i> Activity Logs
                    </a>
                    <hr>
                    <a class="nav-link text-white" href="/">
                        <i class="fas fa-globe"></i> Public Site
                    </a>
                    <a class="nav-link text-white" href="/logout">
                        <i class="fas fa-sign-out-alt"></i> Logout
                    </a>
                </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 content p-4">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h1><i class="fas fa-clipboard-list"></i> Admin Activity Logs</h1>
                    <div class="d-flex gap-2">
                        <select class="form-select" id="filterAction">
                            <option value="">All Actions</option>
                            <option value="admin_login">Admin Login</option>
                            <option value="user_status_update">User Status Update</option>
                            <option value="opportunity_moderate">Opportunity Moderation</option>
                        </select>
                        <input type="date" class="form-control" id="filterDate">
                    </div>
                </div>
                
                <!-- Activity Logs -->
                <div class="card border-0 shadow-sm">
                    <div class="card-header bg-warning text-dark">
                        <h5><i class="fas fa-history"></i> Recent Activity ({{ logs|length }} entries)</h5>
                    </div>
                    <div class="card-body p-0">
                        <div style="max-height: 700px; overflow-y: auto;">
                            {% for log in logs %}
                            <div class="log-entry p-3 mb-2 mx-2 rounded {% if 'login' in log.action %}log-success{% elif 'error' in log.action or 'ban' in log.action %}log-critical{% elif 'warning' in log.action %}log-warning{% endif %}"
                                 data-action="{{ log.action }}" data-date="{{ log.created_at[:10] }}">
                                <div class="d-flex justify-content-between align-items-start">
                                    <div>
                                        <h6 class="mb-1">
                                            <i class="fas fa-user-shield text-primary"></i>
                                            <strong>{{ log.admin_name }}</strong>
                                            <span class="badge bg-secondary ms-2">{{ log.action }}</span>
                                        </h6>
                                        {% if log.details %}
                                        <p class="mb-2 text-muted">{{ log.details }}</p>
                                        {% endif %}
                                        {% if log.target_type and log.target_id %}
                                        <small class="text-info">
                                            <i class="fas fa-bullseye"></i> Target: {{ log.target_type }} (ID: {{ log.target_id[:8] }}...)
                                        </small>
                                        {% endif %}
                                    </div>
                                    <div class="text-end">
                                        <small class="text-muted">
                                            <i class="fas fa-clock"></i> {{ log.created_at }}
                                        </small>
                                        {% if log.ip_address %}
                                        <br><small class="text-muted">
                                            <i class="fas fa-globe"></i> {{ log.ip_address }}
                                        </small>
                                        {% endif %}
                                    </div>
                                </div>
                            </div>
                            {% endfor %}
                            
                            {% if not logs %}
                            <div class="text-center py-5">
                                <i class="fas fa-clipboard-list fa-3x text-muted mb-3"></i>
                                <h5 class="text-muted">No activity logs found</h5>
                                <p class="text-muted">Admin activities will be logged here.</p>
                            </div>
                            {% endif %}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Filter functionality
        document.getElementById('filterAction').addEventListener('change', function(e) {
            const filterAction = e.target.value;
            const entries = document.querySelectorAll('.log-entry');
            
            entries.forEach(entry => {
                const entryAction = entry.dataset.action;
                entry.style.display = (!filterAction || entryAction === filterAction) ? 'block' : 'none';
            });
        });
        
        document.getElementById('filterDate').addEventListener('change', function(e) {
            const filterDate = e.target.value;
            const entries = document.querySelectorAll('.log-entry');
            
            entries.forEach(entry => {
                const entryDate = entry.dataset.date;
                entry.style.display = (!filterDate || entryDate === filterDate) ? 'block' : 'none';
            });
        });
    </script>
</body>
</html>
'''