<?php

namespace Kambo\Langchain\Tests\LLMs;

use Closure;
use Exception;
use GuzzleHttp\Client as GuzzleClient;
use GuzzleHttp\Handler\MockHandler;
use GuzzleHttp\HandlerStack;
use GuzzleHttp\Psr7\Response;
use Http\Discovery\Psr18Client;
use Kambo\Langchain\LLMs\Enums\OpenAIChatModel;
use Kambo\Langchain\LLMs\OpenAIChat;
use OpenAI\Client;
use OpenAI\Transporters\HttpTransporter;
use OpenAI\ValueObjects\ApiKey;
use OpenAI\ValueObjects\Transporter\BaseUri;
use OpenAI\ValueObjects\Transporter\Headers;
use OpenAI\ValueObjects\Transporter\QueryParams;
use PHPUnit\Framework\TestCase;
use Psr\Http\Client\ClientInterface;
use Psr\Http\Message\RequestInterface;
use Psr\Http\Message\ResponseInterface;

use function json_encode;
use function array_merge;
use function is_null;

class OpenAIChatTest extends TestCase
{
    /**
     * @dataProvider provideModelData
     */
    public function testExecute($inputModel, $resultingModel): void
    {
        $openAI = $this->mockOpenAIWithResponses(
            [
                self::prepareResponse(
                    [
                        'id' => 'chatcmpl-6yGpmeZ6v6cALFWagesgA9zvaYNTs',
                        'object' => 'chat.completion',
                        'created' => 1679822410,
                        'model' => $resultingModel,
                        'choices' =>
                            [
                                0 =>
                                    [
                                        'index' => 0,
                                        'message' =>
                                            [
                                                'role' => 'assistant',
                                                'content' => 'Happy Feet Co.',
                                            ],
                                        'finish_reason' => 'stop',
                                    ],
                            ],
                        'usage' =>
                            [
                                'prompt_tokens' => 23,
                                'completion_tokens' => 4,
                                'total_tokens' => 27,
                            ],
                    ]
                )
            ],
            [
                'model_name' => $inputModel
            ]
        );

        $this->assertEquals(
            'Happy Feet Co.',
            $openAI('What would be a good company name for a company that makes colorful socks?')
        );
    }

    /**
     * @dataProvider provideModelData
     */
    public function testToArray($inputModel, $resultingModel): void
    {
        $openAI = $this->mockOpenAIWithResponses([], [
            'model_name' => $inputModel
        ]);

        $this->assertEquals(
            [
                'model_name' => $resultingModel,
                'model_kwargs' => [],
            ],
            $openAI->toArray(),
        );
    }

    /**
     * @dataProvider provideModelData
     */
    public function testGenerate($inputModel, $resultingModel): void
    {
        $openAI = $this->mockOpenAIWithResponses(
            [
                self::prepareResponse(
                    [
                        'id' => 'chatcmpl-6yGpmeZ6v6cALFWagesgA9zvaYNTs',
                        'object' => 'chat.completion',
                        'created' => 1679822410,
                        'model' => $resultingModel,
                        'choices' =>
                            [
                                0 =>
                                    [
                                        'index' => 0,
                                        'message' =>
                                            [
                                                'role' => 'assistant',
                                                'content' => 'Happy Feet Co.',
                                            ],
                                        'finish_reason' => 'stop',
                                    ],
                            ],
                        'usage' =>
                            [
                                'prompt_tokens' => 23,
                                'completion_tokens' => 4,
                                'total_tokens' => 27,
                            ],
                    ]
                )
            ],
            [
                'model_name' => $inputModel
            ]
        );

        $result = $openAI->generate(['Tell me a joke']);

        $this->assertEquals(
            'Happy Feet Co.',
            $result->getFirstGenerationText()
        );

        $answers = [];
        foreach ($result->getGenerations() as $generation) {
            foreach ($generation as $gen) {
                $answers[] = $gen->text;
            }
        }

        $this->assertEquals(
            [
                'Happy Feet Co.',
            ],
            $answers
        );

        $this->assertEquals(
            [
                'token_usage' => [
                    'prompt_tokens' => 23,
                    'completion_tokens' => 4,
                    'total_tokens' => 27,
                ],
            ],
            $result->getLLMOutput()
        );
    }

    private static function prepareResponse(array $response): Response
    {
        return new Response(200, ['Content-Type' => 'application/json'], json_encode($response));
    }

    private static function mockOpenAIWithResponses(array $responses = [], array $options = []): OpenAIChat
    {
        $mock = new MockHandler($responses);

        $client = self::client($mock);
        return new OpenAIChat(array_merge(['openai_api_key' => 'test'], $options), $client);
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

    public function provideModelData()
    {
        return [
            [
                null,
                OpenAIChatModel::Gpt35Turbo->value,
            ],
            [
                OpenAIChatModel::Gpt35Turbo->value,
                OpenAIChatModel::Gpt35Turbo->value,
            ],
            [
                OpenAIChatModel::Gpt4->value,
                OpenAIChatModel::Gpt4->value,
            ]
        ];
    }
}
