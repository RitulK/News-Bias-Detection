import { Inter } from 'next/font/google'
import './globals.css'
import "@fontsource/inter"; // Import the Inter font
import Navbar from '@/components/main/Navbar';

const inter = Inter({ 
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
})

export const metadata = {
  title: 'News Bias Detection',
  description: 'Analyze news bias with AI',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <Navbar />
      <body className={`${inter.variable} font-sans`}>{children}</body>
    </html>
  )
}
