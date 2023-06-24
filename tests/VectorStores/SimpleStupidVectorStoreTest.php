<?php

namespace Kambo\Langchain\Tests\VectorStores;

use Closure;
use Exception;
use GuzzleHttp\Client as GuzzleClient;
use GuzzleHttp\Handler\MockHandler;
use GuzzleHttp\HandlerStack;
use GuzzleHttp\Psr7\Response;
use Http\Discovery\Psr18Client;
use Kambo\Langchain\Embeddings\OpenAIEmbeddings;
use OpenAI\Client;
use OpenAI\Transporters\HttpTransporter;
use OpenAI\ValueObjects\ApiKey;
use OpenAI\ValueObjects\Transporter\BaseUri;
use OpenAI\ValueObjects\Transporter\Headers;
use OpenAI\ValueObjects\Transporter\QueryParams;
use PHPUnit\Framework\TestCase;
use Kambo\Langchain\VectorStores\SimpleStupidVectorStore;
use Kambo\Langchain\Docstore\Document;
use Psr\Http\Client\ClientInterface;
use Psr\Http\Message\RequestInterface;
use Psr\Http\Message\ResponseInterface;

use function json_encode;
use function array_merge;
use function is_null;

class SimpleStupidVectorStoreTest extends TestCase
{
    public function testEmbedDocuments(): void
    {
        $openAI = $this->mockOpenAIEmbeddingsWithResponses(
            [
                self::prepareResponse(
                    [
                        'object' => 'list',
                        'data' => [
                            [
                                'object' => 'embedding',
                                'index' => 0,
                                'embedding' =>
                                    [
                                        -0.015587599,
                                        -0.03145355,
                                        -0.010950541,
                                        -0.014322372,
                                        -0.0121335285,
                                        -0.0009655265,
                                        -0.025747374,
                                        0.0009908311,
                                        -0.017751137,
                                        -0.010210384,
                                        0.0010643724,
                                    ],
                            ],
                        ],
                        'usage' => [
                            'prompt_tokens' => 1468,
                            'total_tokens' => 1468,
                        ],
                    ]
                ),
                self::prepareResponse(
                    [
                        'object' => 'list',
                        'data' => [
                            [
                                'object' => 'embedding',
                                'index' => 0,
                                'embedding' =>
                                    [
                                        -0.015587599,
                                        -0.03145355,
                                        -0.010950541,
                                        -0.014322372,
                                        -0.0121335285,
                                        -0.0009655265,
                                        -0.025747374,
                                        0.0009908311,
                                        -0.017751137,
                                        -0.010210384,
                                        0.0010643724,
                                    ],
                            ],
                        ],
                        'usage' => [
                            'prompt_tokens' => 1468,
                            'total_tokens' => 1468,
                        ],
                    ]
                )
            ]
        );

        $SSVS = new SimpleStupidVectorStore($openAI);
        $SSVS->addTexts(['foo bar baz'], []);

        $this->assertEquals(
            [
                new Document('foo bar baz'),
            ],
            $SSVS->similaritySearch('foo bar baz', 1)
        );
    }

    private static function prepareResponse(array $response): Response
    {
        return new Response(200, ['Content-Type' => 'application/json'], json_encode($response));
    }

    private static function mockOpenAIEmbeddingsWithResponses(array $responses, array $options = []): OpenAIEmbeddings
    {
        $mock = new MockHandler($responses);

        $client = self::client($mock);
        return new OpenAIEmbeddings(array_merge(['openai_api_key' => 'test'], $options), $client);
    }

    private static function client(MockHandler $mockHandler): Client
    {
        $apiKey = ApiKey::from('test');
        $baseUri = BaseUri::from('api.openai.com/v1');
        $headers = Headers::withAuthorization($apiKey);

        $handlerStack = HandlerStack::create($mockHandler);
        $client = new GuzzleClient(['handler' => $handlerStack]);

        $queryParams = QueryParams::create();
        $sendAsync = self::makeStreamHandler($client);

        $transporter = new HttpTransporter($client, $baseUri, $headers, $queryParams, $sendAsync);

        return new Client($transporter);
    }



    private static ?Closure $streamHandler = null;

    /**
     * Creates a new stream handler for "stream" requests.
     */
    private static function makeStreamHandler(ClientInterface $client): Closure
    {
        if (! is_null(self::$streamHandler)) {
            return self::$streamHandler;
        }

        if ($client instanceof GuzzleClient) {
            return fn (RequestInterface $request): ResponseInterface => $client->send($request, ['stream' => true]);
        }

        if ($client instanceof Psr18Client) { // @phpstan-ignore-line
            return fn (RequestInterface $request): ResponseInterface => $client->sendRequest($request); // @phpstan-ignore-line
        }

        return function (RequestInterface $_): never {
            throw new Exception('To use stream requests you must provide an stream handler closure via the OpenAI factory.');
        };
    }
}
