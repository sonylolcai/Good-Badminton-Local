'use client'
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Database, LayoutDashboard, LogOut, MapPin, Server, Settings, Users } from 'lucide-react';
import { cn } from '@/lib/utils';

export function Sidebar({ platformAdmin, username, onLogout }: { platformAdmin: boolean; username: string; onLogout: () => void }) {
  const pathname = usePathname();

  const links = [
    { name: '运营概览', href: '/', icon: LayoutDashboard, platformOnly: true },
    { name: '球馆与场地', href: '/venues', icon: MapPin },
    { name: '用户与球员', href: '/users', icon: Users },
    { name: '视频资源', href: '/resources', icon: Database },
    { name: 'GPU 服务', href: '/gpu', icon: Server, platformOnly: true },
    { name: '系统设置', href: '/settings', icon: Settings, platformOnly: true },
  ];

  return (
    <div className="w-64 bg-slate-900 text-white flex flex-col h-full shadow-xl z-10">
      <div className="p-6 text-center border-b border-slate-800">
        <h1 className="text-xl font-bold tracking-wider text-indigo-400">Good Badminton</h1>
      </div>
      <nav className="flex-1 px-4 py-6 space-y-2">
        {links.filter((link) => !link.platformOnly || platformAdmin).map((link) => {
          const isActive = pathname === link.href || (link.href !== '/' && pathname.startsWith(`${link.href}/`));
          const Icon = link.icon;
          return (
            <Link
              key={link.name}
              href={link.href}
              className={cn(
                "flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors",
                isActive ? "bg-indigo-600 text-white shadow-md" : "text-slate-300 hover:bg-slate-800 hover:text-white"
              )}
            >
              <Icon className="w-5 h-5" />
              <span className="font-medium">{link.name}</span>
            </Link>
          );
        })}
      </nav>
      <div className="border-t border-slate-800 p-4"><p className="truncate px-2 text-sm text-slate-300">{username}</p><button type="button" onClick={onLogout} className="mt-2 flex w-full items-center gap-2 rounded-lg px-2 py-2 text-sm text-slate-300 hover:bg-slate-800 hover:text-white"><LogOut className="h-4 w-4"/>退出登录</button></div>
    </div>
  );
}
