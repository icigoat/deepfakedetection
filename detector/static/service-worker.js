// Service Worker for AI Media Detector PWA
const CACHE_NAME = 'ai-detector-v1';
const RUNTIME_CACHE = 'ai-detector-runtime';

// Files to cache on install
const STATIC_ASSETS = [
  '/',
  '/static/manifest.json',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css',
  'https://cdn.jsdelivr.net/npm/chart.js'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('[Service Worker] Installing...');
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[Service Worker] Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => self.skipWaiting())
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[Service Worker] Activating...');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME && name !== RUNTIME_CACHE)
          .map((name) => {
            console.log('[Service Worker] Deleting old cache:', name);
            return caches.delete(name);
          })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch event - network first, then cache
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip cross-origin requests
  if (url.origin !== location.origin) {
    // For CDN resources, use cache first
    if (url.hostname.includes('cdnjs.cloudflare.com') || url.hostname.includes('cdn.jsdelivr.net')) {
      event.respondWith(
        caches.match(request).then((cachedResponse) => {
          return cachedResponse || fetch(request).then((response) => {
            return caches.open(RUNTIME_CACHE).then((cache) => {
              cache.put(request, response.clone());
              return response;
            });
          });
        })
      );
    }
    return;
  }

  // For API calls (analyze endpoint), always use network
  if (url.pathname.includes('/analyze/')) {
    event.respondWith(fetch(request));
    return;
  }

  // For media files, use network only
  if (url.pathname.includes('/media/')) {
    event.respondWith(fetch(request));
    return;
  }

  // For everything else, try network first, fall back to cache
  event.respondWith(
    fetch(request)
      .then((response) => {
        // Clone the response
        const responseClone = response.clone();
        
        // Cache successful responses
        if (response.status === 200) {
          caches.open(RUNTIME_CACHE).then((cache) => {
            cache.put(request, responseClone);
          });
        }
        
        return response;
      })
      .catch(() => {
        // If network fails, try cache
        return caches.match(request).then((cachedResponse) => {
          if (cachedResponse) {
            return cachedResponse;
          }
          
          // If no cache, return offline page
          return new Response(
            '<html><body><h1>Offline</h1><p>You are currently offline. Please check your internet connection.</p></body></html>',
            {
              headers: { 'Content-Type': 'text/html' }
            }
          );
        });
      })
  );
});

// Handle messages from clients
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
