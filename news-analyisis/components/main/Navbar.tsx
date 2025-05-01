"use client"
import { Button } from "@/components/ui/button";
import { Menu, Search, ChevronRight } from "lucide-react";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Input } from "@/components/ui/input";

export default function Navbar() {
  return (
    <nav className="flex items-center justify-between px-6 py-4 bg-[rgba(255, 255, 255, 0.50)] backdrop-blur-xl text-white border-b border-white/10 shadow-sm rounded-b-4xl fixed top-0 left-0 w-full z-50">
      {/* Logo */}
      <div className="flex items-center gap-2">
        <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center">
          <span className="font-bold text-white text-sm">NN</span>
        </div>
        <div className="text-lg font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-indigo-500">
          NewsNation
        </div>
      </div>
      
      {/* Desktop Navigation (Centered) */}
      <div className="hidden md:flex gap-8 absolute left-1/2 transform -translate-x-1/2">
        <a 
          key="Home" 
          href="/"
          className="text-blue-600 hover:text-white transition-all duration-200 text-sm font-medium relative px-3 py-1.5 rounded-md group"
        >
          <span className="relative z-10">Home</span>
          <span className="absolute inset-0 bg-white/10 rounded-md opacity-0 group-hover:opacity-100 transition-all duration-300"></span>
          <span className="absolute -bottom-1 left-0 right-0 h-0.5 bg-gradient-to-r from-blue-400 to-indigo-500 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-300 origin-left"></span>
        </a>
        <a 
          key="Sources" 
          href="/sources"
          className="text-blue-600 hover:text-white transition-all duration-200 text-sm font-medium relative px-3 py-1.5 rounded-md group"
        >
          <span className="relative z-10">Sources</span>
          <span className="absolute inset-0 bg-white/10 rounded-md opacity-0 group-hover:opacity-100 transition-all duration-300"></span>
          <span className="absolute -bottom-1 left-0 right-0 h-0.5 bg-gradient-to-r from-blue-400 to-indigo-500 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-300 origin-left"></span>
        </a>
        <a 
          key="Methodology" 
          href="/methodology"
          className="text-blue-600 hover:text-white transition-all duration-200 text-sm font-medium relative px-3 py-1.5 rounded-md group"
        >
          <span className="relative z-10">Methodology</span>
          <span className="absolute inset-0 bg-white/10 rounded-md opacity-0 group-hover:opacity-100 transition-all duration-300"></span>
          <span className="absolute -bottom-1 left-0 right-0 h-0.5 bg-gradient-to-r from-blue-400 to-indigo-500 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-300 origin-left"></span>
        </a>
      </div>
      
      {/* Sleek Search Bar */}
      <div className="hidden md:flex items-center px-3 py-2 rounded-md border border-blue-700/30 transition-all duration-300 hover:border-blue-300/40 w-56 h-10">
        <Search size={16} className="text-blue-700 mr-2" />
        <Input 
          placeholder="Search..." 
          className="bg-transparent border-none shadow-none focus-visible:ring-0 focus-visible:ring-offset-0 text-blue-500 placeholder-blue-500 text-sm h-full p-0" 
        />
        <Button 
          size="sm" 
          className="ml-auto h-6 w-6 rounded-full bg-blue-500/40 hover:bg-blue-500/60 text-white p-0 flex items-center justify-center transition-colors"
        >
          <ChevronRight size={14} />
        </Button>
      </div>
      
      {/* Mobile Menu */}
      <Sheet>
        <SheetTrigger asChild>
          <Button variant="ghost" size="icon" className="md:hidden text-white hover:bg-white/10">
            <Menu size={24} />
          </Button>
        </SheetTrigger>
        <SheetContent side="left" className="bg-gradient-to-br from-slate-900/95 to-indigo-900/95 backdrop-blur-xl text-white w-64 p-6 border-r border-white/10">
          <div className="flex items-center gap-2 mb-8">
            <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center">
              <span className="font-bold text-white text-sm">NN</span>
            </div>
            <div className="text-lg font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400">
              NewsNation
            </div>
          </div>
          
          <nav className="flex flex-col gap-6 text-sm">
            {['Home', 'Sources', 'Methodology', 'Contact'].map((item, index) => (
              <a 
                key={item}
                href="#" 
                className="text-gray-300 hover:text-white transition-all duration-200 flex items-center group"
              >
                <div className="h-1.5 w-1.5 rounded-full bg-blue-500 mr-3 opacity-60 group-hover:opacity-100 transition-all duration-300"></div>
                <span>{item}</span>
              </a>
            ))}
          </nav>
          
          <div className="mt-8 pt-8 border-t border-white/10">
            <div className="flex items-center gap-2 bg-white/10 backdrop-blur-md px-3 py-2 rounded-full border border-white/20 shadow-inner shadow-black/5">
              <Search size={15} className="text-blue-300" />
              <Input 
                placeholder="Search..." 
                className="bg-transparent border-none focus-visible:ring-0 focus-visible:ring-offset-0 text-white placeholder-gray-400 text-xs flex-1 h-6 px-0" 
              />
            </div>
          </div>
        </SheetContent>
      </Sheet>
    </nav>
  );
}
