using System.Runtime.Serialization.Formatters;
using AI_project_escapeRoom; // Replace 'YourNamespace' with the actual namespace where Game1 is defined
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using System.Linq;
class GameEnvironment
{
    private Game1 game;
    private int maxSteps = 500;
    private int currentStep;

    public GameEnvironment(Game1 game)
    {
        this.game = game;
        this.currentStep = 0;
    }

    public double[] GetState()
    {
        // Example state representation
        return new double[]
        {
            game.player.Position.X / game.widthLevel,
            game.player.Position.Y / game.groundLevel,
            game.box.Position.X / game.widthLevel,
            game.box.Position.Y / game.groundLevel,
            game.IsPressed ? 1.0 : 0.0,
            game.IsOpen ? 1.0 : 0.0
        };
    }

    public (double[], int, bool) Step(int action)
    {
        currentStep++;
        int reward = 0;
        bool IsDone = false;
        // Apply action
        switch (action)
        {
            case 0: game.player.Move(new Vector2(-10, 0)); break; // Move left
            case 1: game.player.Move(new Vector2(10, 0)); break;  // Move right
            case 2: if (game.player.IsGrounded) game.player.ApplyForce(new Vector2(0, -250)); break; // Jump
            case 3: game.player.Grab(game.box); break; // Interact (example)
            case 4: game.player.DropHeldBox(); break; // Interact (example)
        }

        // Check for rewards
        if (game.IsPressed)
            reward += 10;
        if (game.IsOpen)
            reward += 100;
        if (game.player.Intersects(game.box))
            reward += 5;

        // Check if the game is over
        if (game.IsOpen || currentStep >= maxSteps)
            reward += -1; // Penalty for taking too long

        if (game.player.Position.X < game.cameraPosition.X || game.player.Position.X > game.cameraPosition.X + game.ScreenWidth ||
            game.player.Position.Y < game.cameraPosition.Y || game.player.Position.Y > game.cameraPosition.Y + game.ScreenHeight)
            IsDone = true; // Penalty for going out of bounds
        return (GetState(), reward, IsDone);
    }

    public bool IsDone()
    {
        return game.IsOpen || currentStep >= maxSteps;
    }
}