const http = require('http');

const requestListener = (request, response) => {
    response.setHeader('Content-Type', 'text/html');
    response.statusCode = 200;

    const { method, url } = request;

    console.log(`Received request: ${method} ${url}`);

    if (url === '/') {
        if (method === 'GET') {
            response.end('<h1>Ini adalah homepage</h1>');
        } else {
            response.end(`<h1>Halaman tidak dapat diakses dengan ${method} request</h1>`);
        }
    } else if (url === '/about') {
        if (method === 'GET') {
            response.end('<h1>Halo! Ini adalah halaman about</h1>');
        } else if (method === 'POST') {
            let body = [];

            request.on('data', (chunk) => {
                body.push(chunk);
            });

            request.on('end', () => {
                body = Buffer.concat(body).toString();
                console.log('Raw body received:', body);  // Log the raw body received

                try {
                    const parsedBody = JSON.parse(body);  // Try parsing the body as JSON
                    console.log('Parsed body:', parsedBody);

                    const { name } = parsedBody;
                    response.end(`<h1>Halo, ${name}! Ini adalah halaman about</h1>`);
                } catch (err) {
                    console.error('Error parsing JSON:', err);  // Log any JSON parsing errors
                    response.statusCode = 400;
                    response.end(`<h1>Invalid JSON: ${err.message}</h1>`);
                }
            });
        } else {
            response.end(`<h1>Halaman tidak dapat diakses menggunakan ${method} request</h1>`);
        }
    } else {
        response.end('<h1>Halaman tidak ditemukan!</h1>');
    }
};

const server = http.createServer(requestListener);

const port = 5000;
const host = 'localhost';

server.listen(port, host, () => {
    console.log(`Server berjalan pada http://${host}:${port}`);
});
