import { useEffect, useState, type ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import NotFound from '@/pages/not-found';
import { Route, Switch, useLocation, Router as WouterRouter } from 'wouter';
import AppShell from '@/components/layout/AppShell';
import { Dashboard, HelpPage, MeetingDetailsPage, MeetingsPage, NewMeetingPage, ProcessingPage, SettingsPage } from '@/pages/Workspace';

const queryClient = new QueryClient();
function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}>{children}</ErrorBoundary>;
}
function Router() {
  const [dark, setDark] = useState(() => localStorage.getItem('swbs-theme') === 'dark');
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
    localStorage.setItem('swbs-theme', dark ? 'dark' : 'light');
  }, [dark]);
  return <RoutedErrorBoundary><AppShell><Switch>
    <Route path="/" component={Dashboard}/>
    <Route path="/meetings" component={MeetingsPage}/>
    <Route path="/meetings/:id" component={MeetingDetailsPage}/>
    <Route path="/new-meeting" component={NewMeetingPage}/>
    <Route path="/processing/:id" component={ProcessingPage}/>
    <Route path="/settings">{() => <SettingsPage dark={dark} setDark={setDark}/>}</Route>
    <Route path="/help" component={HelpPage}/>
    <Route component={NotFound}/>
  </Switch></AppShell></RoutedErrorBoundary>;
}
function App() {
  return <QueryClientProvider client={queryClient}><TooltipProvider><WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}><Router/></WouterRouter><Toaster/></TooltipProvider></QueryClientProvider>;
}
export default App;
