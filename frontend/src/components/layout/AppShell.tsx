import { useState, type ReactNode } from 'react';
import { Link, useLocation } from 'wouter';
import { Activity, Archive, ChevronLeft, ChevronRight, CircleHelp, LayoutDashboard, Menu, Settings, Upload, X } from 'lucide-react';

const nav = [
  { href: '/', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/meetings', label: 'Meetings', icon: Archive },
  { href: '/new-meeting', label: 'New meeting', icon: Upload },
];
export default function AppShell({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const active = (href: string) => href === '/' ? location === '/' : location.startsWith(href);
  const Sidebar = () => <aside className={`sidebar sidebar-transition ${collapsed ? 'sidebar-collapsed' : ''} ${mobileOpen ? 'sidebar-mobile-open' : ''}`}>
    <div className="brand">
      {!collapsed && <div className="brand-copy"><strong>SMART WAY</strong><span>BUSINESS SOLUTIONS</span></div>}
      <button className="icon-button sidebar-collapse" onClick={() => setCollapsed(!collapsed)} aria-label={collapsed ? 'Expand navigation' : 'Collapse navigation'} data-testid="button-collapse-sidebar">{collapsed ? <ChevronRight size={17}/> : <ChevronLeft size={17}/>}</button>
      <button className="icon-button mobile-close" onClick={() => setMobileOpen(false)} aria-label="Close navigation"><X size={18}/></button>
    </div>
    <div className="nav-label">{!collapsed && 'WORKSPACE'}</div>
    <nav aria-label="Main navigation">{nav.map(item => { const Icon = item.icon; return <Link href={item.href} key={item.href} onClick={() => setMobileOpen(false)} className={`nav-item ${active(item.href) ? 'active' : ''}`} data-testid={`link-nav-${item.label.toLowerCase().replace(' ','-')}`} aria-current={active(item.href) ? 'page' : undefined}><Icon size={18}/>{!collapsed && <span>{item.label}</span>}{item.href === '/meetings' && !collapsed && <span className="nav-count">24</span>}</Link>; })}</nav>
    <div className="sidebar-spacer"/>
    <div className="nav-label">{!collapsed && 'PREFERENCES'}</div>
    <Link href="/settings" onClick={() => setMobileOpen(false)} className={`nav-item ${active('/settings') ? 'active' : ''}`} data-testid="link-nav-settings"><Settings size={18}/>{!collapsed && <span>Settings</span>}</Link>
    <Link href="/help" onClick={() => setMobileOpen(false)} className={`nav-item ${active('/help') ? 'active' : ''}`} data-testid="link-nav-help"><CircleHelp size={18}/>{!collapsed && <span>Help & support</span>}</Link>
    <div className="profile">
      <div className="profile-avatar">AM</div>{!collapsed && <div className="profile-meta"><strong>Amal Mansour</strong><span>Operations</span></div>}
      {!collapsed && <span className="online-dot" title="Available"/>}
    </div>
  </aside>;
  return <div className="app-shell">
    <Sidebar/>
    {mobileOpen && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)}/>}
    <div className={`main-frame ${collapsed ? 'frame-collapsed' : ''}`}>
      <header className="topbar">
        <button className="icon-button mobile-menu" aria-label="Open navigation" onClick={() => setMobileOpen(true)} data-testid="button-open-navigation"><Menu size={20}/></button>
        <div className="topbar-context"><Activity size={15}/><span>Meeting Intelligence</span><span className="context-divider">/</span><span className="context-muted">Internal workspace</span></div>
        <div className="topbar-right"><span className="demo-pill"><i/> Demo workspace</span><span className="topbar-avatar">AM</span></div>
      </header>
      <main className="main-content">{children}</main>
      <footer className="footer"><span>SMART WAY BUSINESS SOLUTIONS</span><span>Meeting Intelligence · Internal tool</span></footer>
    </div>
  </div>;
}
