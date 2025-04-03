"use client"
import { Button } from "@/components/ui/button";
import { Menu, Search } from "lucide-react";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Input } from "@/components/ui/input";

export default function Navbar() {
  return (
    <nav className="flex items-center justify-between px-6 py-3 bg-black/90 backdrop-blur-md text-white shadow-md fixed top-0 left-0 w-full z-50">
      {/* Logo */}
      <div className="text-lg font-bold tracking-wide ml-4">NewsNation</div>
      
      {/* Desktop Navigation (Centered) */}
      <div className="hidden md:flex gap-6 text-gray-300 text-sm absolute left-1/2 transform -translate-x-1/2">
        <a href="#" className="hover:text-white transition">Home</a>
        <a href="#" className="hover:text-white transition">Sources</a>
        <a href="#" className="hover:text-white transition">Methodolgy</a>
      </div>
      
      {/* Sleek Search Bar */}
      <div className="hidden md:flex items-center gap-2 bg-white px-3 py-1.5 rounded-full border border-black shadow-sm w-64">
        <Search size={16} className="text-black" />
        <Input placeholder="Search..." className="bg-transparent border-none focus:ring-0 text-black placeholder-gray-600 text-sm flex-1" />
      </div>
      
      {/* Mobile Menu */}
      <Sheet>
        <SheetTrigger asChild>
          <Button variant="ghost" size="icon" className="md:hidden text-white">
            <Menu size={24} />
          </Button>
        </SheetTrigger>
        <SheetContent side="left" className="bg-black/90 text-white w-64 p-6">
          <nav className="flex flex-col gap-4 text-sm">
            <a href="#" className="hover:text-gray-400 transition">Home</a>
            <a href="#" className="hover:text-gray-400 transition">Features</a>
            <a href="#" className="hover:text-gray-400 transition">Pricing</a>
            <a href="#" className="hover:text-gray-400 transition">Contact</a>
          </nav>
        </SheetContent>
      </Sheet>
    </nav>
  );
}
