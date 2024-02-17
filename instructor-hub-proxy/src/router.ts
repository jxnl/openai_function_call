import { Router } from 'itty-router';
import YAML from 'yaml';

// now let's create a router (note the lack of "new")
const router = Router();

// Function to track analytics
async function trackAnalytics(request: any, env: Env, event_type: string, slug: string, branch: string) {
	const user_agent = request.headers.get('User-Agent') || 'unknown';
	const request_ip = request.headers.get('CF-Connecting-IP') || 'unknown'; // Cloudflare passes the client IP
	const request_time = new Date().toISOString();

	// Prepare and execute the insert statement for analytics tracking
	// @ts-ignore
	await env.DB.prepare(
		'INSERT INTO hub_analytics (event_type, user_agent, request_ip, request_time, slug, branch) VALUES (?, ?, ?, ?, ?, ?)'
	)
		.bind(event_type, user_agent, request_ip, request_time, slug, branch)
		.run();
}

// GET collection index
router.get('/api/:branch/items', async (request) => {
	const { params, env } = request;
	await trackAnalytics(request, env, 'COLLECTION_INDEX', 'index', params.branch);

	const url = `https://raw.githubusercontent.com/jxnl/instructor/${params.branch}/mkdocs.yml?raw=true`;
	const mkdoc_yml = await fetch(url).then((res) => res.text());
	var mkdocs = YAML.parse(mkdoc_yml);
	const cookbooks = mkdocs.nav
		?.filter((obj: Map<string, string>) => 'Hub' in obj)[0]
		.Hub.map((obj: any, index: number) => {
			const [name, path] = Object.entries(obj)[0];
			// Extract slug by getting the substring after the last '/'
			// @ts-ignore
			const slug = path.substring(path.lastIndexOf('/') + 1, path.lastIndexOf('.'));
			return { id: index, name, path, slug };
		})
		.filter(({ slug }: any) => slug !== 'index');

	return new Response(JSON.stringify(cookbooks), {
		headers: {
			'content-type': 'application/json',
		},
	});
});

// GET content
router.get('/api/:branch/items/:slug/md', async (request) => {
	const { params, env } = request;
	await trackAnalytics(request, env, 'CONTENT_MARKDOWN', params.slug, params.branch);
	const raw_content = `https://raw.githubusercontent.com/jxnl/instructor/${params.branch}/docs/hub/${params.slug}.md?raw=true`;
	const content = await fetch(raw_content).then((res) => res.text());

	return new Response(content, {
		headers: {
			'content-type': 'text/plain',
		},
	});
});

// GET content python
router.get('/api/:branch/items/:slug/py', async (request) => {
	const { params, env } = request;
	await trackAnalytics(request, env, 'CONTENT_PYTHON', params.slug, params.branch);
	const raw_content = `https://raw.githubusercontent.com/jxnl/instructor/${params.branch}/docs/hub/${params.slug}.md?raw=true`;
	const content = await fetch(raw_content).then((res) => res.text());

	// Extract all Python code blocks from within ```py or ```python blocks in the markdown
	const python_codes = content.match(/(?<=```(?:py|python)\n)[\s\S]+?(?=\n```)/g);

	if (python_codes === null) {
		return new Response('No Python code found in this document.', {
			headers: {
				'content-type': 'text/plain',
			},
		});
	}

	if (python_codes.length === 0) {
		return new Response('No Python code found in this document.', {
			headers: {
				'content-type': 'text/plain',
			},
		});
	}

	const python_code = python_codes.join('\n\n');

	return new Response(python_code, {
		headers: {
			'content-type': 'text/plain',
		},
	});
});

// 404 for everything else
router.all('*', () => new Response('Not Found.', { status: 404 }));

export default router;
