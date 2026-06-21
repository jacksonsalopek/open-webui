<script lang="ts">
	import { getRAGConfig, updateRAGConfig } from '$lib/apis/retrieval';
	import Switch from '$lib/components/common/Switch.svelte';

	import { onMount, getContext } from 'svelte';
	import SensitiveInput from '$lib/components/common/SensitiveInput.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';

	const i18n = getContext('i18n');

	export let saveHandler: Function;

	// This fork only ships Kagi as a search engine; backend support for other
	// providers has been stripped, so the engine selector is gone too — the
	// value is pinned in the env (WEB_SEARCH_ENGINE=kagi).
	let webLoaderEngines = ['playwright', 'firecrawl', 'tavily', 'external'];

	let webConfig = null;

	const submitHandler = async () => {
		// Convert domain filter string to array before sending
		if (
			typeof webConfig.WEB_SEARCH_DOMAIN_FILTER_LIST === 'string' &&
			webConfig.WEB_SEARCH_DOMAIN_FILTER_LIST
		) {
			webConfig.WEB_SEARCH_DOMAIN_FILTER_LIST = webConfig.WEB_SEARCH_DOMAIN_FILTER_LIST.split(',')
				.map((domain) => domain.trim())
				.filter((domain) => domain.length > 0);
		} else if (!Array.isArray(webConfig.WEB_SEARCH_DOMAIN_FILTER_LIST)) {
			webConfig.WEB_SEARCH_DOMAIN_FILTER_LIST = [];
		}

		// Convert Youtube loader language string to array before sending
		if (
			typeof webConfig.YOUTUBE_LOADER_LANGUAGE === 'string' &&
			webConfig.YOUTUBE_LOADER_LANGUAGE
		) {
			webConfig.YOUTUBE_LOADER_LANGUAGE = webConfig.YOUTUBE_LOADER_LANGUAGE.split(',')
				.map((lang) => lang.trim())
				.filter((lang) => lang.length > 0);
		} else if (!Array.isArray(webConfig.YOUTUBE_LOADER_LANGUAGE)) {
			webConfig.YOUTUBE_LOADER_LANGUAGE = [];
		}

		// Convert numeric timeout values to strings (backend expects strings)
		if (typeof webConfig.FIRECRAWL_TIMEOUT === 'number') {
			webConfig.FIRECRAWL_TIMEOUT = webConfig.FIRECRAWL_TIMEOUT.toString();
		}
		if (typeof webConfig.PLAYWRIGHT_TIMEOUT === 'number') {
			webConfig.PLAYWRIGHT_TIMEOUT = webConfig.PLAYWRIGHT_TIMEOUT.toString();
		}

		const res = await updateRAGConfig(localStorage.token, {
			web: { ...webConfig }
		});

		// Convert arrays back to strings for display
		if (Array.isArray(webConfig.WEB_SEARCH_DOMAIN_FILTER_LIST)) {
			webConfig.WEB_SEARCH_DOMAIN_FILTER_LIST = webConfig.WEB_SEARCH_DOMAIN_FILTER_LIST.join(',');
		}
		if (Array.isArray(webConfig.YOUTUBE_LOADER_LANGUAGE)) {
			webConfig.YOUTUBE_LOADER_LANGUAGE = webConfig.YOUTUBE_LOADER_LANGUAGE.join(',');
		}
	};

	onMount(async () => {
		const res = await getRAGConfig(localStorage.token);

		if (res) {
			webConfig = res.web;

			// Convert array back to comma-separated string for display
			if (Array.isArray(webConfig?.WEB_SEARCH_DOMAIN_FILTER_LIST)) {
				webConfig.WEB_SEARCH_DOMAIN_FILTER_LIST = webConfig.WEB_SEARCH_DOMAIN_FILTER_LIST.join(',');
			} else if (!webConfig.WEB_SEARCH_DOMAIN_FILTER_LIST) {
				webConfig.WEB_SEARCH_DOMAIN_FILTER_LIST = '';
			}

			if (Array.isArray(webConfig?.YOUTUBE_LOADER_LANGUAGE)) {
				webConfig.YOUTUBE_LOADER_LANGUAGE = webConfig.YOUTUBE_LOADER_LANGUAGE.join(',');
			} else if (!webConfig.YOUTUBE_LOADER_LANGUAGE) {
				webConfig.YOUTUBE_LOADER_LANGUAGE = '';
			}

			// Convert timeout strings to numbers for number input fields
			if (webConfig.FIRECRAWL_TIMEOUT && typeof webConfig.FIRECRAWL_TIMEOUT === 'string') {
				const parsed = parseInt(webConfig.FIRECRAWL_TIMEOUT);
				if (!Number.isNaN(parsed)) {
					webConfig.FIRECRAWL_TIMEOUT = parsed;
				}
			}
			if (webConfig.PLAYWRIGHT_TIMEOUT && typeof webConfig.PLAYWRIGHT_TIMEOUT === 'string') {
				const parsed = parseInt(webConfig.PLAYWRIGHT_TIMEOUT);
				if (!Number.isNaN(parsed)) {
					webConfig.PLAYWRIGHT_TIMEOUT = parsed;
				}
			}
		}
	});
</script>

<form
	class="flex flex-col h-full justify-between space-y-3 text-sm"
	on:submit|preventDefault={async () => {
		await submitHandler();
		saveHandler();
	}}
>
	<div class=" space-y-3 overflow-y-scroll scrollbar-hidden h-full">
		{#if webConfig}
			<div class="">
				<div class="mb-3">
					<div class=" mt-0.5 mb-2.5 text-base font-medium">{$i18n.t('General')}</div>

					<hr class=" border-gray-100/30 dark:border-gray-850/30 my-2" />

					<div class="  mb-2.5 flex w-full justify-between">
						<div class=" self-center text-xs font-medium">
							{$i18n.t('Web Search')}
						</div>
						<div class="flex items-center relative">
							<Switch bind:state={webConfig.ENABLE_WEB_SEARCH} />
						</div>
					</div>

					<div class="mb-2.5 flex w-full flex-col">
						<div>
							<div class=" self-center text-xs font-medium mb-1">
								{$i18n.t('Kagi Search API Key')}
							</div>

							<SensitiveInput
								placeholder={$i18n.t('Enter Kagi Search API Key')}
								bind:value={webConfig.KAGI_SEARCH_API_KEY}
							/>
						</div>
					</div>

					{#if webConfig.ENABLE_WEB_SEARCH}
						<div class="mb-2.5 flex w-full flex-col">
							<div class="flex gap-2">
								<div class="w-full">
									<div class=" self-center text-xs font-medium mb-1">
										{$i18n.t('Search Result Count')}
									</div>

									<input
										class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
										placeholder={$i18n.t('Search Result Count')}
										bind:value={webConfig.WEB_SEARCH_RESULT_COUNT}
										required
									/>
								</div>

								<div class="w-full">
									<div class=" self-center text-xs font-medium mb-1">
										<Tooltip
											content={$i18n.t(
												'Limit concurrent search queries. 0 = unlimited (default). Set to 1 for sequential execution (recommended for APIs with strict rate limits).'
											)}
											placement="top-start"
										>
											{$i18n.t('Concurrent Requests')}
										</Tooltip>
									</div>

									<input
										class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
										placeholder={$i18n.t('Concurrent Requests')}
										bind:value={webConfig.WEB_SEARCH_CONCURRENT_REQUESTS}
										type="number"
										min="0"
									/>
								</div>
							</div>
						</div>

						<div class="mb-2.5 w-full">
							<div class=" self-center text-xs font-medium mb-1">
								<Tooltip
									content={$i18n.t(
										'Maximum characters to return from fetched URLs. Leave empty for no limit.'
									)}
									placement="top-start"
								>
									{$i18n.t('Fetch URL Content Length Limit')}
								</Tooltip>
							</div>

							<input
								class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
								placeholder={$i18n.t('No limit')}
								bind:value={webConfig.WEB_FETCH_MAX_CONTENT_LENGTH}
								type="number"
								min="0"
							/>
						</div>

						<div class="mb-2.5 flex w-full flex-col">
							<div class="  text-xs font-medium mb-1">
								{$i18n.t('Domain Filter List')}
							</div>

							<input
								class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
								placeholder={$i18n.t(
									'Enter domains separated by commas (e.g., example.com,site.org,!excludedsite.com)'
								)}
								bind:value={webConfig.WEB_SEARCH_DOMAIN_FILTER_LIST}
							/>
						</div>
					{/if}

					<div class="  mb-2.5 flex w-full justify-between">
						<div class=" self-center text-xs font-medium">
							<Tooltip content={$i18n.t('Full Context Mode')} placement="top-start">
								{$i18n.t('Bypass Embedding and Retrieval')}
							</Tooltip>
						</div>
						<div class="flex items-center relative">
							<Tooltip
								content={webConfig.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL
									? $i18n.t(
											'Inject the entire content as context for comprehensive processing, this is recommended for complex queries.'
										)
									: $i18n.t(
											'Default to segmented retrieval for focused and relevant content extraction, this is recommended for most cases.'
										)}
							>
								<Switch bind:state={webConfig.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL} />
							</Tooltip>
						</div>
					</div>

					<div class="  mb-2.5 flex w-full justify-between">
						<div class=" self-center text-xs font-medium">
							<Tooltip content={$i18n.t('Bypass Web Loader')} placement="top-start">
								{$i18n.t('Bypass Web Loader')}
							</Tooltip>
						</div>
						<div class="flex items-center relative">
							<Tooltip content={''}>
								<Switch bind:state={webConfig.BYPASS_WEB_SEARCH_WEB_LOADER} />
							</Tooltip>
						</div>
					</div>

					<div class="  mb-2.5 flex w-full justify-between">
						<div class=" self-center text-xs font-medium">
							{$i18n.t('Trust Proxy Environment')}
						</div>
						<div class="flex items-center relative">
							<Tooltip
								content={webConfig.WEB_SEARCH_TRUST_ENV
									? $i18n.t(
											'Use proxy designated by http_proxy and https_proxy environment variables to fetch page contents.'
										)
									: $i18n.t('Use no proxy to fetch page contents.')}
							>
								<Switch bind:state={webConfig.WEB_SEARCH_TRUST_ENV} />
							</Tooltip>
						</div>
					</div>
				</div>

				<div class="mb-3">
					<div class=" mt-0.5 mb-2.5 text-base font-medium">{$i18n.t('Loader')}</div>

					<hr class=" border-gray-100/30 dark:border-gray-850/30 my-2" />

					<div class="  mb-2.5 flex w-full justify-between">
						<div class=" self-center text-xs font-medium">
							{$i18n.t('Web Loader Engine')}
						</div>
						<div class="flex items-center relative">
							<select
								class="w-fit pr-8 rounded-sm px-2 p-1 text-xs bg-transparent outline-hidden text-right"
								bind:value={webConfig.WEB_LOADER_ENGINE}
								placeholder={$i18n.t('Select a engine')}
							>
								<option value="">{$i18n.t('Default')}</option>
								{#each webLoaderEngines as engine}
									<option value={engine}>{engine}</option>
								{/each}
							</select>
						</div>
					</div>

					{#if webConfig.WEB_LOADER_ENGINE === '' || webConfig.WEB_LOADER_ENGINE === 'safe_web'}
						<div class="  mb-2.5 flex w-full justify-between">
							<div class=" self-center text-xs font-medium">
								{$i18n.t('Timeout')}
							</div>
							<div class="flex items-center relative">
								<input
									class="flex-1 w-full text-sm bg-transparent outline-hidden"
									placeholder={$i18n.t('Timeout')}
									bind:value={webConfig.WEB_LOADER_TIMEOUT}
								/>
							</div>
						</div>

						<div class="  mb-2.5 flex w-full justify-between">
							<div class=" self-center text-xs font-medium">
								{$i18n.t('Verify SSL Certificate')}
							</div>
							<div class="flex items-center relative">
								<Switch bind:state={webConfig.ENABLE_WEB_LOADER_SSL_VERIFICATION} />
							</div>
						</div>
					{:else if webConfig.WEB_LOADER_ENGINE === 'playwright'}
						<div class="mb-2.5 flex w-full flex-col">
							<div>
								<div class=" self-center text-xs font-medium mb-1">
									{$i18n.t('Playwright WebSocket URL')}
								</div>

								<div class="flex w-full">
									<div class="flex-1">
										<input
											class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
											type="text"
											placeholder={$i18n.t('Enter Playwright WebSocket URL')}
											bind:value={webConfig.PLAYWRIGHT_WS_URL}
											autocomplete="off"
										/>
									</div>
								</div>
							</div>

							<div class="mt-2">
								<div class=" self-center text-xs font-medium mb-1">
									{$i18n.t('Playwright Timeout (ms)')}
								</div>

								<div class="flex w-full">
									<div class="flex-1">
										<input
											class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
											placeholder={$i18n.t('Enter Playwright Timeout')}
											bind:value={webConfig.PLAYWRIGHT_TIMEOUT}
											autocomplete="off"
										/>
									</div>
								</div>
							</div>
						</div>
					{:else if webConfig.WEB_LOADER_ENGINE === 'firecrawl'}
						<div class="mb-2.5 flex w-full flex-col">
							<div>
								<div class=" self-center text-xs font-medium mb-1">
									{$i18n.t('Firecrawl API Base URL')}
								</div>

								<div class="flex w-full">
									<div class="flex-1">
										<input
											class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
											type="text"
											placeholder={$i18n.t('Enter Firecrawl API Base URL')}
											bind:value={webConfig.FIRECRAWL_API_BASE_URL}
											autocomplete="off"
										/>
									</div>
								</div>
							</div>

							<div class="mt-2">
								<div class=" self-center text-xs font-medium mb-1">
									{$i18n.t('Firecrawl API Key')}
								</div>

								<SensitiveInput
									placeholder={$i18n.t('Enter Firecrawl API Key')}
									bind:value={webConfig.FIRECRAWL_API_KEY}
								/>
							</div>
						</div>
					{:else if webConfig.WEB_LOADER_ENGINE === 'tavily'}
						<div class="mb-2.5 flex w-full flex-col">
							<div>
								<div class=" self-center text-xs font-medium mb-1">
									{$i18n.t('Tavily Extract Depth')}
								</div>

								<div class="flex w-full">
									<div class="flex-1">
										<input
											class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
											type="text"
											placeholder={$i18n.t('Enter Tavily Extract Depth')}
											bind:value={webConfig.TAVILY_EXTRACT_DEPTH}
											autocomplete="off"
										/>
									</div>
								</div>
							</div>

							<div class="mt-2">
								<div class=" self-center text-xs font-medium mb-1">
									{$i18n.t('Tavily API Key')}
								</div>

								<SensitiveInput
									placeholder={$i18n.t('Enter Tavily API Key')}
									bind:value={webConfig.TAVILY_API_KEY}
								/>
							</div>
						</div>
					{:else if webConfig.WEB_LOADER_ENGINE === 'external'}
						<div class="mb-2.5 flex w-full flex-col">
							<div>
								<div class=" self-center text-xs font-medium mb-1">
									{$i18n.t('External Web Loader URL')}
								</div>

								<div class="flex w-full">
									<div class="flex-1">
										<input
											class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
											type="text"
											placeholder={$i18n.t('Enter External Web Loader URL')}
											bind:value={webConfig.EXTERNAL_WEB_LOADER_URL}
											autocomplete="off"
										/>
									</div>
								</div>
							</div>

							<div class="mt-2">
								<div class=" self-center text-xs font-medium mb-1">
									{$i18n.t('External Web Loader API Key')}
								</div>

								<SensitiveInput
									placeholder={$i18n.t('Enter External Web Loader API Key')}
									bind:value={webConfig.EXTERNAL_WEB_LOADER_API_KEY}
								/>
							</div>
						</div>
					{/if}

					<div class="mb-2.5 w-full">
						<div class=" self-center text-xs font-medium mb-1">
							{$i18n.t('Concurrent Requests')}
						</div>

						<input
							class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
							placeholder={$i18n.t('Concurrent Requests')}
							bind:value={webConfig.WEB_LOADER_CONCURRENT_REQUESTS}
							required
						/>
					</div>

					<div class="  mb-2.5 flex w-full justify-between">
						<div class=" self-center text-xs font-medium">
							{$i18n.t('Youtube Language')}
						</div>
						<div class="flex items-center relative">
							<input
								class="flex-1 w-full rounded-lg text-sm bg-transparent outline-hidden"
								type="text"
								placeholder={$i18n.t('Enter language codes')}
								bind:value={webConfig.YOUTUBE_LOADER_LANGUAGE}
								autocomplete="off"
							/>
						</div>
					</div>

					<div class="  mb-2.5 flex flex-col w-full justify-between">
						<div class=" mb-1 text-xs font-medium">
							{$i18n.t('Youtube Proxy URL')}
						</div>
						<div class="flex items-center relative">
							<input
								class="flex-1 w-full rounded-lg text-sm bg-transparent outline-hidden"
								type="text"
								placeholder={$i18n.t('Enter proxy URL (e.g. https://user:password@host:port)')}
								bind:value={webConfig.YOUTUBE_LOADER_PROXY_URL}
								autocomplete="off"
							/>
						</div>
					</div>
				</div>
			</div>
		{/if}
	</div>
	<div class="flex justify-end pt-3 text-sm font-medium">
		<button
			class="px-3.5 py-1.5 text-sm font-medium bg-black hover:bg-gray-900 text-white dark:bg-white dark:text-black dark:hover:bg-gray-100 transition rounded-full"
			type="submit"
		>
			{$i18n.t('Save')}
		</button>
	</div>
</form>
